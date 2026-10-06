#include <sys/types.h>
#include <unistd.h>

/*
 * Jinn labels the task user as "myuser", but some images map that account to
 * UID 0. Unreal Engine refuses to start when libc reports UID 0. This shim is
 * loaded only into the CARLA server process and reports an unprivileged UID;
 * it does not change the process' kernel credentials.
 */
/* Use an account that exists in the image's passwd database. */
static const uid_t carla_uid = 65534;

uid_t getuid(void) { return carla_uid; }
uid_t geteuid(void) { return carla_uid; }

int getresuid(uid_t *ruid, uid_t *euid, uid_t *suid) {
  if (ruid != NULL) {
    *ruid = carla_uid;
  }
  if (euid != NULL) {
    *euid = carla_uid;
  }
  if (suid != NULL) {
    *suid = carla_uid;
  }
  return 0;
}
