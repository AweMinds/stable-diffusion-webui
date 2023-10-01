import argparse
import os
import time
import logging

logging.basicConfig(filename='./logs/start.log',
                    level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d): %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", help="端口")
parser.add_argument("-d", "--device-id", help="GPU的序号")
args = parser.parse_args()

logging.info("port: " + args.port)
logging.info("device-id: " + args.device_id)

SYSTEM_MONITOR_ADDR = os.environ.get('SYSTEM_MONITOR_ADDR')
SYSTEM_MONITOR_API_SECRET = os.environ.get('SYSTEM_MONITOR_API_SECRET')
START_ARGS = os.environ.get("START_ARGS")

n = 0
while True:
    logging.info(f"[Device {args.device_id}] -  Relauncher: Launching...")

    if n > 0:
        logging.info(f'\tRelaunch count: {n}')

    launch_string = f"./webui.sh -f --port {args.port} --device-id {args.device_id}"

    if not SYSTEM_MONITOR_ADDR:
        pass
    else:
        launch_string += f" --system-monitor-addr {SYSTEM_MONITOR_ADDR}" \
                         f" --system-monitor-api-secret {SYSTEM_MONITOR_API_SECRET}"

    if not START_ARGS:
        pass
    else:
        launch_string += f" {START_ARGS}"

    logging.info(f"[Device {args.device_id}] - launch_string: {launch_string}")
    os.system(launch_string)
    logging.info(f"[Device {args.device_id}] - Relauncher: Process is ending. Relaunching in 2s...")
    n += 1
    time.sleep(2)
