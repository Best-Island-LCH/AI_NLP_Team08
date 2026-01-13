#!/usr/bin/env python
"""GPU 할당 테스트 스크립트"""
import os
import subprocess
import time

def test_gpu(gpu_id):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    result = subprocess.run(
        ['python', '-c', '''
import torch
print(f"CUDA_VISIBLE_DEVICES={__import__("os").environ.get("CUDA_VISIBLE_DEVICES", "not set")}")
print(f"cuda.is_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"current_device={torch.cuda.current_device()}")
    print(f"device_name={torch.cuda.get_device_name(0)}")
'''],
        env=env,
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr

print("=" * 50)
print("GPU 0 테스트")
print("=" * 50)
out, err = test_gpu(0)
print(out)
if err:
    print(f"Error: {err}")

print("=" * 50)
print("GPU 1 테스트")
print("=" * 50)
out, err = test_gpu(1)
print(out)
if err:
    print(f"Error: {err}")

print("=" * 50)
print("병렬 실행 테스트")
print("=" * 50)

# 병렬로 실행
env0 = os.environ.copy()
env0['CUDA_VISIBLE_DEVICES'] = '0'
env1 = os.environ.copy()
env1['CUDA_VISIBLE_DEVICES'] = '1'

p0 = subprocess.Popen(
    ['python', '-c', 'import torch; import time; print(f"GPU0: {torch.cuda.get_device_name(0)}"); time.sleep(2); print("GPU0 완료")'],
    env=env0, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)
p1 = subprocess.Popen(
    ['python', '-c', 'import torch; import time; print(f"GPU1: {torch.cuda.get_device_name(0)}"); time.sleep(2); print("GPU1 완료")'],
    env=env1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)

out0, err0 = p0.communicate()
out1, err1 = p1.communicate()

print(f"GPU 0 결과:\n{out0}")
print(f"GPU 1 결과:\n{out1}")

if err0:
    print(f"GPU 0 에러: {err0}")
if err1:
    print(f"GPU 1 에러: {err1}")

print("\n✅ 테스트 완료!")
