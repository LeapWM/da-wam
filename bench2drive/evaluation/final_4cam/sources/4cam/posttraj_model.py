"""Original JQTF forward with final trajectory-conditioned future tokens."""
from navsim.agents.drive_jepa_perception_based.drive_jepa_model import *

class PostTrajModel(DriveJEPAModel):
    @torch.no_grad()
    def encode_future_target(self, camera_input: torch.Tensor) -> torch.Tensor:
        """Return the original NAVSIM PostTraj 256-D EMA target map."""
        self._ema_target_backbone.eval()
        return self._ema_target_backbone(camera_input, img_metas={})[4]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        from multiview import encode_four_cameras
        camera_feature, cam_f_1, image_feature_with_map = encode_four_cameras(self, features)
        ego_status = features['ego_status'][:, -1].clone()
        batch_size = ego_status.shape[0]
        if self.b2d:
            ego_status[:, 1:3] = 0

        projected_map = image_feature_with_map[4]  # [B, 256, 16, 32]
        lora_vit_map = image_feature_with_map[5]  # [B, 1024, 16, 32], grad to LoRA
        image_feature = image_feature_with_map[:4]   # 4-tuple, backward compatible for refiner

        output={}
        if self._ema_jqtf_enabled:
            # The target cache supplies t+0.5.  Retaining normalized t lets the
            # EMA teacher receive the required [t,t+0.5] clip without a cache
            # format change.  Validation loss needs the same contract; NavTest
            # does not contain future targets and therefore never runs teacher.
            output["future_target_current_frame"] = cam_f_1.detach()

        predicted_futures = None
        future_context_latent = None
        scorer_context_latent = None
        if (
            self._config.future_integration in ("scorer_pool", "ema_jqtf")
            and self._should_run_future_predictor()
        ):
            if getattr(self._config, "future_gt_trajectory_conditioned", False):
                scorer_context_latent = self._future_context_latent(
                    camera_feature, projected_map, lora_vit_map
                )
                future_context_latent = projected_map
                output["future_context_latent"] = future_context_latent
                output["scorer_context_latent"] = scorer_context_latent
            else:
                predicted_futures = self._predict_future_latents(
                    camera_feature, projected_map, lora_vit_map
                )
                output["predicted_future_latents"] = predicted_futures

        target_point=features['target_point'].float()
        target_valid=features['target_point_valid'].float().reshape(batch_size,1)
        if target_point.shape!=(batch_size,2):
            raise ValueError(f'Expected target_point [B,2], got {tuple(target_point.shape)}')
        route_feature=self.route_target_encoder(target_point)*target_valid
        # Give every proposal access to all cameras before its first decode.
        # Near-origin initial proposals can lie outside the front frustum, in
        # which case the geometric refiner has no visible samples yet.
        scene_feature=projected_map.mean(dim=(-2,-1))
        ego_feature=(self.hist_encoding(ego_status)+route_feature+scene_feature)[:,None]

        bev_feature =ego_feature+self.init_feature.weight[None]

        proposal_list = []
        joint_query_feature = None

        for i, refine in enumerate(self._trajectory_head):
            if self._ema_jqtf_enabled:
                # On the final iteration this is Q3: the exact pre-decode query
                # that produces proposal^3.  The returned feature is Q4.
                joint_query_feature = bev_feature
            bev_feature, proposal_list = refine(bev_feature, proposal_list,image_feature)

        proposals=proposal_list[-1]

        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        candidate_future_tokens = None
        if self._ema_jqtf_enabled:
            if (
                future_context_latent is None
                or scorer_context_latent is None
                or joint_query_feature is None
            ):
                raise RuntimeError(
                    "EMA-JQTF requires current latent and final pre-decode Q3"
                )
            joint_queries = bev_feature.reshape(
                batch_size,
                self._config.proposal_num,
                self.poses_num,
                self._config.tf_d_model,
            )
            candidate_future_tokens = (
                self._future_predictor.predict_candidate_tokens(
                    future_context_latent, proposals.detach(), joint_queries
                )
            )
            output["joint_query_features"] = joint_queries
            output["candidate_future_tokens"] = candidate_future_tokens
            if self.training:
                output['collision_object_prediction']=self.collision_object_head(proposals,joint_queries,candidate_future_tokens)
            output["projected_map"] = projected_map

        if self.training and getattr(self._config, "scorer_anchor_auxiliary", False):
            # Training-only retained state. Keep the full pose dimension for
            # nearest-slot gather now and future current-encoder fusion later.
            output["proposal_pose_features"] = bev_feature.reshape(
                batch_size,
                self._config.proposal_num,
                self.poses_num,
                self._config.tf_d_model,
            )
            output["projected_map"] = projected_map

        future_latent_for_scorer = None
        proposal_future_features = None
        if self._config.future_integration == "scorer_pool":
            if predicted_futures is not None:
                future_latent_for_scorer = self._apply_future_diag(
                    predicted_futures[0], projected_map, lora_vit_map
                )
            elif future_context_latent is not None:
                proposal_future_features = self.predict_proposal_rollout_features(
                    future_context_latent, proposals
                )
                proposal_future_features = self._apply_proposal_rollout_diag(
                    proposal_future_features
                )
                if proposal_future_features is not None:
                    output["proposal_future_features"] = proposal_future_features

        if self._ema_jqtf_enabled:
            gradient_mode = self._config.ema_jqtf_scorer_gradient_mode
            if gradient_mode == "detached":
                scorer_trajectories = proposals.detach()
                scorer_pose_features = bev_feature.detach()
                scorer_scene = projected_map.detach()
                scorer_future = candidate_future_tokens.detach()
                scorer_context = scorer_context_latent.detach()
            elif gradient_mode == "representation":
                # The label is evaluated on the current proposal coordinates.
                # Do not use that fixed-label gradient to directly move the
                # coordinates, but let score supervision shape the generated
                # query, scene representation, and future representation.
                scorer_trajectories = proposals.detach()
                scorer_pose_features = bev_feature
                scorer_scene = projected_map
                scorer_future = candidate_future_tokens
                scorer_context = scorer_context_latent
            elif gradient_mode == "full":
                scorer_trajectories = proposals
                scorer_pose_features = bev_feature
                scorer_scene = projected_map
                scorer_future = candidate_future_tokens
                scorer_context = scorer_context_latent
            else:
                raise RuntimeError(
                    "invalid EMA-JQTF scorer gradient mode at forward: "
                    f"{gradient_mode!r}"
                )
            (
                pred_logit,
                pred_logit2,
                pred_agents_states,
                pred_area_logit,
                bev_semantic_map,
                agent_states,
                agent_labels,
                pred_pdm_progress_norm,
            ) = self.scorer(
                scorer_trajectories,
                scorer_pose_features,
                scorer_scene,
                scorer_future,
                scorer_context,
            )
        else:
            pred_pdm_progress_norm = None
            (
                pred_logit,
                pred_logit2,
                pred_agents_states,
                pred_area_logit,
                bev_semantic_map,
                agent_states,
                agent_labels,
            ) = self.scorer(
                proposals,
                bev_feature,
                future_latent_for_scorer,
                projected_map,
                proposal_future_features,
            )

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"]=agent_states
        output["agent_labels"]=agent_labels
        output["pred_pdm_progress_norm"] = pred_pdm_progress_norm

        selection_score = None
        if (
            self._ema_jqtf_enabled
            and self._config.ema_jqtf_scorer_head_mode
            == "formula_progress"
        ):
            if pred_pdm_progress_norm is None:
                raise RuntimeError(
                    "formula_progress scorer did not return PDM progress"
                )
            metric_scores = torch.sigmoid(pred_logit[..., :4])
            noc = metric_scores[..., 0]
            dac = metric_scores[..., 1]
            ttc = metric_scores[..., 2]
            comfort = metric_scores[..., 3]
            raw_progress_norm = pred_logit[..., 4]
            formula_selection = formula_progress_selection(
                metric_scores,
                raw_progress_norm,
                pred_pdm_progress_norm,
                progress_scale=float(self._config.scorer_progress_scale),
                mode=self._config.scorer_formula_selection_mode,
                safety_topk=int(
                    self._config.scorer_formula_safety_topk
                ),
                progress_mode=self._config.scorer_formula_progress_mode,
                progress_distance_threshold=float(
                    self._config.scorer_progress_distance_threshold
                ),
                safe_progress_tie_break_weight=float(
                    self._config.scorer_formula_safe_progress_tie_break_weight
                ),
            )
            ep = formula_selection.ego_progress
            pdm_score = formula_selection.formula_score
            selection_score = formula_selection.selection_score
            progress_scale = float(
                self._config.scorer_progress_scale
            )
            output["pred_metric_scores"] = metric_scores
            output["pred_raw_progress_norm"] = raw_progress_norm
            output["pred_raw_progress"] = (
                raw_progress_norm * progress_scale
            )
            output["pred_pdm_progress"] = (
                pred_pdm_progress_norm * progress_scale
            )
            output["pred_ep"] = ep
            output["pred_formula_safety_rank"] = (
                formula_selection.safety_rank
            )
        elif self._config.scorer_traj_conditioned and self._config.scorer_pdm_formula_inference:
            # PDM-like reward from factorized subscore heads [NOC,DAC,EP,TTC,comfort]
            s = torch.sigmoid(pred_logit)
            noc, dac, ep, ttc, comfort = s[..., 0], s[..., 1], s[..., 2], s[..., 3], s[..., 4]
            pdm_score = noc * dac * (5.0 * ttc + 5.0 * ep + 2.0 * comfort) / 12.0
        elif pred_logit2 is not None:
            pdm_score=(torch.sigmoid(pred_logit)+torch.sigmoid(pred_logit2))[:,:,-1]/2
        else:
            pdm_score=torch.sigmoid(pred_logit)[:,:,-1]

        if selection_score is None:
            selection_score = pdm_score
        token = torch.argmax(selection_score, dim=1)
        batch_index = torch.arange(batch_size, device=proposals.device)
        trajectory = proposals[batch_index, token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score
        output["selection_score"] = selection_score
        output["selected_index"] = token

        if (
            self._ema_jqtf_enabled
            and not self.training
            and self._config.ema_jqtf_decode_selected_future
        ):
            output["selected_future_latent"] = (
                self._future_predictor.decode_full_latent(
                    future_context_latent,
                    self._future_predictor._gather_candidates(
                        candidate_future_tokens, token
                    ),
                )
            )

        if (
            predicted_futures is None
            and future_context_latent is None
            and self._should_run_future_predictor()
        ):
            predicted_futures = self._predict_future_latents(
                camera_feature, projected_map, lora_vit_map
            )
            output["predicted_future_latents"] = predicted_futures

        return output
