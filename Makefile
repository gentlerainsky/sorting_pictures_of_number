build:
	docker -l debug build -t local/ml_project .

shell:
	docker run --shm-size 16G --gpus all -it -p 8800:8800 -p 3123:3123 -v `pwd`:/workspace local/ml_project

shell-more:
	docker exec -it $$(docker ps | awk 'FNR==2{print $$1}') /bin/bash

jupyter:
	jupyter notebook --ip 0.0.0.0 --port=8800 --no-browser --allow-root

