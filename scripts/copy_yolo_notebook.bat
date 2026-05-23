@echo off
set SRC=C:\Users\moonjintae\projects\face_attr_mtl\notebooks\yolo_train_and_tune.ipynb
set DST=C:\Users\moonjintae\projects\emotion bot(cursor ver)\notebooks\yolo\yolo_train_and_tune.ipynb
mkdir "%~dp0..\notebooks\yolo" 2>nul
copy /Y "%SRC%" "%DST%"
echo copied to %DST%
