# Final weights

Only the final models and their required pretrained backbone are retained:

- `navsim_final.ckpt`: SafetyFinal epoch 1, shared by NAVSIM v1 and v2.
- `bench2drive_4cam_score11.ckpt`: final four-camera epoch 11.
- `vjepa2_1_vitl.pt`: required V-JEPA 2.1 pretrained backbone.

All three are regular files. Other checkpoints and external training references were moved to the sibling `leap-auto-wam-backup` folder. They are not required for inference. Training Stage 2 requires explicitly supplying a Stage 1 checkpoint via `CHECKPOINT_PATH`.

See `manifest.json` for sizes and `final_sha256.json` for hashes. Hard-linked archived copies share content; never overwrite a retained checkpoint in place.
