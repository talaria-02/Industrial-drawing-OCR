# [파일 설명]
# 윈도우 환경에서 PPOCRLabel을 실행할 때 발생하는 PyQt5와 PyTorch 간의 DLL 충돌 버그를
# 우회하기 위해 작성된 래퍼 스크립트입니다. UI를 띄우기 전에 torch를 
# 먼저 임포트하여 안전하게 실행합니다.

"""
PPOCRLabel 실행 래퍼 — Windows PyQt5+PyTorch DLL 충돌 우회
=============================================================
증상: `PPOCRLabel --lang en` 실행 시 검은 콘솔창만 깜빡이고 죽음.

원인: PPOCRLabel.py는 (GUI 앱이라) PyQt5를 먼저 import하고, 그 다음
paddleocr → paddlex → modelscope → torch 순으로 나중에 import한다.
Windows에서 PyQt5가 먼저 프로세스의 DLL 검색 경로를 건드려 놓으면,
뒤늦게 로드되는 torch의 네이티브 DLL(c10.dll) 초기화가 실패한다
(OSError: WinError 1114). paddleocr 자체는 우리 프로젝트 전체에서
문제없이 잘 돌아가고 있었음 — torch는 우리가 쓰지도 않는데 paddlex의
선택적 의존성(modelscope) 경로를 통해 딸려 들어오는 것이고, 이게
PyQt5보다 늦게 로드되는 게 문제의 전부.

해결: 이 스크립트에서 PyQt5가 로드되기 전에 torch를 먼저 한 번 import해
DLL을 정상 적재해 둔다. 이후 같은 프로세스 안에서 import torch가 다시
호출돼도(PPOCRLabel.py 내부에서) sys.modules 캐시를 그대로 반환하므로
초기화 코드가 재실행되지 않는다 — 이후 순서와 무관하게 안전해짐.

사용법 (repo 루트에서):
  python finetune/run_ppocrlabel.py --lang en
"""

import sys

try:
    import torch  # noqa: F401  (사용 안 함 — DLL 선적재만이 목적)
except ImportError:
    pass  # torch가 아예 없다면 애초에 문제도 없음

from PPOCRLabel.PPOCRLabel import main

if __name__ == '__main__':
    main()
