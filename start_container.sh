#!/bin/bash
echo "Container Started"
export PYTHONUNBUFFERED=1
#source /venv/bin/activate
#
#echo "syncing venv to workspace, please wait"
#rsync -au --remove-source-files /venv/ /workspace/venv/
#
#echo "syncing stable diffusion to workspace, please wait"
#rsync -au --remove-source-files /stable-diffusion-webui/ /workspace/stable-diffusion-webui/

#START_PORT=7860

if [[ $RUNPOD_STOP_AUTO ]]; then
  echo "Skipping auto-start of webui"
else
  echo "Started webui through relauncher script"
  mkdir -p logs
  port=7860
  device_id=0
  if [[ $START_PORT ]]; then
    port=$START_PORT
  fi
  if [[ $DEVICE_ID ]]; then
    device_id=$DEVICE_ID
  fi

  python relauncher-awe.py --port "${port}" --device-id "${device_id}" &
fi

if [[ $PUBLIC_KEY ]]; then
  mkdir -p ~/.ssh
  chmod 700 ~/.ssh
  cd ~/.ssh
  echo $PUBLIC_KEY >>authorized_keys
  chmod 700 -R ~/.ssh
  cd /
  service ssh start
  echo "SSH Service Started"
fi

if [[ $JUPYTER_PASSWORD ]]; then
  ln -sf /examples /workspace
  ln -sf /root/welcome.ipynb /workspace

  cd /
  jupyter lab --allow-root --no-browser --port=8888 --ip=* \
    --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
    --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=* --ServerApp.preferred_dir=/workspace
  echo "Jupyter Lab Started"
fi

sleep infinity
