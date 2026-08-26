#!/usr/bin/env bash
# Stage 5 — switch EKF2 from GPS to MARIO external-vision velocity.
#
# Every value below is taken from PX4-Autopilot/src/modules/ekf2/module.yaml for this
# v1.15 tree, not from documentation or memory (work rule 4). Run this against a RUNNING
# PX4, then restart it: EKF2_HGT_REF and EKF2_EV_DELAY are reboot_required, and SITL
# persists parameters to rootfs/parameters.bson across restarts.
#
# Usage:  bash scripts/set_stage5_params.sh [restore]
set -euo pipefail

PX4_BIN=/src/gs25122/PX4-Autopilot/build/px4_sitl_default/bin
ROOTFS=/src/gs25122/PX4-Autopilot/build/px4_sitl_default/rootfs

cd "$ROOTFS"

if [[ "${1:-set}" == "restore" ]]; then
  # module.yaml defaults, so a later GPS baseline run is unaffected by this stage.
  "$PX4_BIN/px4-param" set EKF2_GPS_CTRL 7
  "$PX4_BIN/px4-param" set EKF2_EV_CTRL 15
  "$PX4_BIN/px4-param" set EKF2_HGT_REF 1
  "$PX4_BIN/px4-param" set EKF2_BARO_CTRL 1
  "$PX4_BIN/px4-param" set EKF2_EV_DELAY 0
  "$PX4_BIN/px4-param" save
  echo "restored EKF2 defaults -- restart PX4 to apply"
  exit 0
fi

# bitmask 0-15: bit0 lon/lat, bit1 altitude, bit2 3D velocity, bit3 dual-antenna heading.
# 0 disables GNSS aiding entirely.
"$PX4_BIN/px4-param" set EKF2_GPS_CTRL 0

# bitmask 0-15: bit0 horizontal position, bit1 vertical position, bit2 3D velocity,
# bit3 yaw. 4 = bit2 only, so EKF2 fuses MARIO's velocity and nothing else (결정 A);
# leaving the position bits off keeps the filter's own position state.
"$PX4_BIN/px4-param" set EKF2_EV_CTRL 4

# enum: 0 barometric pressure, 1 GPS, 2 range, 3 vision. 0 puts height on the barometer
# (결정 B) so a vertical estimate error cannot drive a climb or a descent.
"$PX4_BIN/px4-param" set EKF2_HGT_REF 0

# boolean: keep barometric height aiding on (this is already the default).
"$PX4_BIN/px4-param" set EKF2_BARO_CTRL 1

# float, milliseconds, range 0-300. Stage 2 measured p50 end-to-end latency 34.6 ms
# (결정 C). p50 rather than p95: EKF2 wants the typical delay, and the p95 tail was
# 4 samples out of 1996.
"$PX4_BIN/px4-param" set EKF2_EV_DELAY 35

# px4-param marks freshly set values unsaved ("*"); SITL only flushes them to
# rootfs/parameters.bson on a clean shutdown, and the restart procedure here uses pkill.
# Without this the reboot_required params would be silently lost across the restart they
# require.
"$PX4_BIN/px4-param" save

echo
echo "--- readback ---"
for p in EKF2_GPS_CTRL EKF2_EV_CTRL EKF2_HGT_REF EKF2_BARO_CTRL EKF2_EV_DELAY; do
  printf "%-16s " "$p"; "$PX4_BIN/px4-param" show "$p" | sed -n '2p'
done
echo
echo "EKF2_HGT_REF and EKF2_EV_DELAY are reboot_required -- restart PX4 now."
