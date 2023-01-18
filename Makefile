.ONESHELL:
SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Type make build-donkeysim-desktop-cpu IMG_NAME=donkeysim_cpu to build a CPU image for Donkeysim running"
	@echo "Type make build-donkeysim-desktop-gpu IMG_NAME=donkeysim_gpu to build a GPU image for Donkeysim running"

.PHONY: build-donkeysim-desktop-cpu
build-donkeysim-desktop-cpu:
	DOCKER_BUILDKIT=1 docker build \
		-f tools/image/Dockerfile.donkeysim \
		--build-arg BASE_IMAGE=ubuntu:20.04 \
		--build-arg PROC=cpu \
		-t donkeysim_cpu .

.PHONY: build-donkeysim-desktop-gpu
build-donkeysim-desktop-gpu:
	DOCKER_BUILDKIT=1 docker build \
		-f tools/image/Dockerfile.donkeysim \
		--build-arg BASE_IMAGE=nvidia/cuda:11.6.2-devel-ubuntu20.04 \
		--build-arg PROC=gpu \
		-t donkeysim_gpu .

.PHONY: start_donkeysim
start_donkeysim:
	@PROC=${PROC}
	docker compose up --no-start
	docker compose start donkeysim_${PROC}
	docker exec -it donkeysim_${PROC}_cont /bin/bash

.PHONY: copy_from_container
copy_from_container:
	@PROC=${PROC}
	@CONT_PATH=${CONT_PATH}
	@LOCAL_PATH=${LOCAL_PATH}
	docker cp donkeysim_${PROC}_cont:${CONT_PATH} ${LOCAL_PATH}
	sudo chown -R ${USER}:${USER} ${LOCAL_PATH}

.PHONY: copy_to_container
copy_to_container:
	@PROC=${PROC}
	@CONT_PATH=${CONT_PATH}
	@LOCAL_PATH=${LOCAL_PATH}
	docker cp ${LOCAL_PATH} donkeysim_${PROC}_cont:${CONT_PATH}

	