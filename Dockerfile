# 基础镜像
FROM ubuntu:22.04

# 设置工作目录
WORKDIR /app

RUN apt update
RUN apt install -y wget git python3 python3-venv python3-pip

## 安装TCMalloc和ControlNet依赖
RUN apt install -y libgl1-mesa-glx libglib2.0-0 google-perftools sox ffmpeg libcairo2 libcairo2-dev

#文件操作
RUN ln -s /usr/bin/python3 /usr/bin/python
#RUN mv config_prd.json config.json
#RUN mv cache_prd.json cache.json

#clone代码
RUN git clone -b aweminds-merge-1.6 https://github.com/AweMinds/stable-diffusion-webui.git .

# 安装webUI依赖
RUN ./webui.sh -f --skip-torch-cuda-test --exit

RUN ln -s /app/venv/lib/python3.10/site-packages/torch/lib/libnvrtc-672ee683.so.11.2 /app/venv/lib/python3.10/site-packages/torch/lib/libnvrtc.so

#更新代码
RUN mkdir -p /tmp/gradio

## 暴露端口
EXPOSE 7860

# 启动应用
ENTRYPOINT ["/bin/sh", "-c", "./start_container.sh"]

#CMD start-awe.sh $GPU_COUNT