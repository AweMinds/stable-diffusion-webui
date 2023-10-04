import json
import logging
import sys
import traceback
from functools import wraps
import html
import time
import gradio

from modules import shared, progress, errors, devices, fifo_lock, util, sd_vae, system_monitor
from modules.system_monitor import MonitorException

from PIL import Image

from modules.timer import Timer

queue_lock = fifo_lock.FIFOLock()

logger = logging.getLogger(__name__)


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return f


def wrap_gradio_gpu_call(func, extra_outputs=None):
    @wraps(func)
    def f(*args, **kwargs):

        # if the first argument is a string that says "task(...)", it is treated as a job id
        if args and type(args[0]) == str and args[0].startswith("task(") and args[0].endswith(")"):
            id_task = args[0]
            progress.add_task_to_queue(id_task)
        else:
            id_task = None

        task_id = None
        status = ''
        time_consumption = {}
        added_at_time = time.time()
        extra_outputs_array = extra_outputs
        if extra_outputs_array is None:
            extra_outputs_array = [None, '', '']

        with queue_lock:
            timer = Timer('gpu_call')

            shared.state.begin(job=id_task)
            # 任务开始
            task_id = system_monitor.on_task(get_request_from_args(func, args), func, *args, **kwargs)
            time_consumption['in_queue'] = time.time() - added_at_time
            progress.start_task(id_task)

            # 任务进行
            try:
                # 判断是否需要切换模型
                _request = get_request_from_args(func, args)
                _check_sd_model(_request)
                timer.record('load_models')

                # 开始GPU生图
                res = func(*args, **kwargs)
                timer.record('inference')
                progress.record_results(id_task, res)

                # 任务完成
                status = 'finished'
                time_consumption.update(timer.records)
                time_consumption['total'] = time.time() - added_at_time
                logger.info(timer.summary())
            # 任务失败
            except MonitorException as e:
                logger.error(f'task {id_task} failed: {e.__str__()}')
                status = 'failed'
                res = extra_outputs_array + [str(e)]
                monitor_addr = shared.cmd_opts.system_monitor_addr

                if monitor_addr:
                    # AWETODO: monitor的错误处理
                    if 399 < e.status_code < 500:
                        err_response_string = e.__repr__()
                        err_resp_json = json.loads(err_response_string)
                        errcode = err_resp_json['message']['errcode']
                        errmsg = err_resp_json['message']['errmsg']
                        raise gradio.Error(f'error code:{errcode},{errmsg}')
                return res
            except Exception as e:
                logger.error(f'task {id_task} failed: {e.__str__()}')
                if isinstance(e, MonitorException):
                    task_failed = False
                status = 'failed'
                traceback.print_tb(e.__traceback__, file=sys.stderr)
                print(e, file=sys.stderr)
                error_message = f'{type(e).__name__}: {e}'
                res = extra_outputs_array + [f"<div class='error'>{html.escape(error_message)}</div>"]
            finally:
                # 任务结束
                progress.finish_task(id_task)
                if task_id:
                    try:
                        if len(res) > 0 and len(res[0]) > 0 and isinstance(res[0][0], Image.Image):
                            # First element in res is gallery
                            image_paths = [item.already_saved_as for item in res[0] if isinstance(item, Image.Image)]
                            log_message = json.dumps([image_paths] + list(res[1:]))
                        else:
                            log_message = json.dumps(res)
                    except Exception as e:
                        log_message = f'Fail to json serialize results: {str(e)}'
                    try:
                        system_monitor.on_task_finished(get_request_from_args(func, args), task_id, status,
                                                        log_message, time_consumption)
                    except Exception as e:
                        logging.warning(f'send task finished event to monitor failed: {str(e)}')

            shared.state.end()

        return res

    return wrap_gradio_call(f, extra_outputs=extra_outputs, add_stats=True)


def wrap_gradio_call(func, extra_outputs=None, add_stats=False):
    @wraps(func)
    def f(*args, extra_outputs_array=extra_outputs, **kwargs):
        run_memmon = shared.opts.memmon_poll_rate > 0 and not shared.mem_mon.disabled and add_stats
        if run_memmon:
            shared.mem_mon.monitor()
        t = time.perf_counter()

        try:
            res = list(func(*args, **kwargs))
        except Exception as e:
            # When printing out our debug argument list,
            # do not print out more than a 100 KB of text
            max_debug_str_len = 131072
            message = "Error completing request"
            arg_str = f"Arguments: {args} {kwargs}"[:max_debug_str_len]
            if len(arg_str) > max_debug_str_len:
                arg_str += f" (Argument list truncated at {max_debug_str_len}/{len(arg_str)} characters)"
            errors.report(f"{message}\n{arg_str}", exc_info=True)

            shared.state.job = ""
            shared.state.job_count = 0

            if extra_outputs_array is None:
                extra_outputs_array = [None, '']

            error_message = f'{type(e).__name__}: {e}'
            res = extra_outputs_array + [f"<div class='error'>{html.escape(error_message)}</div>"]

        devices.torch_gc()

        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.job_count = 0

        if not add_stats:
            return tuple(res)

        elapsed = time.perf_counter() - t
        elapsed_m = int(elapsed // 60)
        elapsed_s = elapsed % 60
        elapsed_text = f"{elapsed_s:.1f} sec."
        if elapsed_m > 0:
            elapsed_text = f"{elapsed_m} min. "+elapsed_text

        if run_memmon:
            mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}
            active_peak = mem_stats['active_peak']
            reserved_peak = mem_stats['reserved_peak']
            sys_peak = mem_stats['system_peak']
            sys_total = mem_stats['total']
            sys_pct = sys_peak/max(sys_total, 1) * 100

            toltip_a = "Active: peak amount of video memory used during generation (excluding cached data)"
            toltip_r = "Reserved: total amout of video memory allocated by the Torch library "
            toltip_sys = "System: peak amout of video memory allocated by all running programs, out of total capacity"

            text_a = f"<abbr title='{toltip_a}'>A</abbr>: <span class='measurement'>{active_peak/1024:.2f} GB</span>"
            text_r = f"<abbr title='{toltip_r}'>R</abbr>: <span class='measurement'>{reserved_peak/1024:.2f} GB</span>"
            text_sys = f"<abbr title='{toltip_sys}'>Sys</abbr>: <span class='measurement'>{sys_peak/1024:.1f}/{sys_total/1024:g} GB</span> ({sys_pct:.1f}%)"

            vram_html = f"<p class='vram'>{text_a}, <wbr>{text_r}, <wbr>{text_sys}</p>"
        else:
            vram_html = ''

        # last item is always HTML
        res[-1] += f"<div class='performance'><p class='time'>Time taken: <wbr><span class='measurement'>{elapsed_text}</span></p>{vram_html}</div>"

        return tuple(res)

    return f


def get_request_from_args(func, args):
    func_name = func.__name__
    if func_name == "txt2img":
        return args[23]

    if func_name == "img2img":
        return args[37]

    if func_name == "run_postprocessing":
        return args[6]

    return None


def _check_sd_model(_request: gradio.Request = None):
    if not _request:
        return

    model_title = util.get_value_from_cookie("sd_model_checkpoint", _request)
    vae_title = util.get_value_from_cookie("sd_vae", _request)

    if not shared.sd_model or shared.sd_model.sd_checkpoint_info.title != model_title:
        import modules.sd_models
        # refresh model, unload it from memory to prevent OOM
        checkpoint = modules.sd_models.get_closet_checkpoint_match(model_title)
        wrap_queued_call(lambda: modules.sd_models.reload_model_weights(info=checkpoint))
    if shared.sd_model:
        vae_file, vae_source = sd_vae.resolve_vae(shared.sd_model.sd_checkpoint_info.filename, vae_title).tuple()
        if sd_vae.loaded_vae_file != vae_file:
            wrap_queued_call(lambda: sd_vae.reload_vae_weights(vae_file=vae_file))
