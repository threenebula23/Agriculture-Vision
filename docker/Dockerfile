FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

RUN mkdir -p /usr/app/
WORKDIR /usr/app/

COPY requirements.txt /usr/app/
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]