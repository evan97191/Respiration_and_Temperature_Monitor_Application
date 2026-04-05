#!/bin/bash
# scripts/setup_udev.sh
# This script configures permissions on the Linux host to allow 
# non-root access (and Docker access) to the UVC thermal camera.

echo "[UDEV Setup] Applying UVC camera permissions rule for VID=1e4e, PID=0100..."

# Create the rule file
RULE='SUBSYSTEM=="usb", ATTR{idVendor}=="1e4e", ATTR{idProduct}=="0100", GROUP="plugdev", MODE="0666"'
echo "$RULE" | sudo tee /etc/udev/rules.d/99-thermal-cam.rules

# Reload the udev rules
echo "[UDEV Setup] Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "------------------------------------------------------------------"
echo "[UDEV Setup] DONE. IMPORTANT: Please UNPLUG and RE-PLUG your thermal camera now for changes to take effect."
echo "------------------------------------------------------------------"
