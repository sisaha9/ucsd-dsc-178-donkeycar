.ONESHELL:
SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Type make build-donkeycar-desktop-cpu IMG_NAME=donkeycar_cpu to build a CPU image for Donkeycar running"
	@echo "Type make build-donkeycar-desktop-gpu IMG_NAME=donkeycar_gpu to build a GPU image for Donkeycar running"

.PHONY: build-donkeycardesktop-cpu
build-donkeycar-desktop-cpu:
	DOCKER_BUILDKIT=1 docker build \
		-f tools/image/Dockerfile \
		--build-arg BASE_IMAGE=ubuntu:20.04 \
		--build-arg PROC=cpu \
		-t donkeycar_cpu .

.PHONY: build-donkeycar-desktop-gpu
build-donkeycar-desktop-gpu:
	DOCKER_BUILDKIT=1 docker build \
		-f tools/image/Dockerfile \
		--build-arg BASE_IMAGE=nvidia/cuda:11.7.1-devel-ubuntu20.04 \
		--build-arg PROC=gpu \
		-t donkeycar_gpu .

.PHONY: start_donkeycar
start_donkeycar:
	@PROC=${PROC}
	docker compose up --force-recreate --no-start
	docker compose start donkeycar_${PROC}
	docker exec -it donkeycar_${PROC}_cont /bin/bash
	docker compose stop

.PHONY: copy_from_container
copy_from_container:
	@PROC=${PROC}
	@CONT_PATH=${CONT_PATH}
	@LOCAL_PATH=${LOCAL_PATH}
	docker cp donkeycar_${PROC}_cont:${CONT_PATH} ${LOCAL_PATH}
	sudo chown -R ${USER}:${USER} ${LOCAL_PATH}

.PHONY: copy_to_container
copy_to_container:
	@PROC=${PROC}
	@CONT_PATH=${CONT_PATH}
	@LOCAL_PATH=${LOCAL_PATH}
	docker cp ${LOCAL_PATH} donkeycar_${PROC}_cont:${CONT_PATH}

	