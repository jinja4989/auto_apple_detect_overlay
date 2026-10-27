# 사과게임 합10 안내기

판을 처음 볼 때 **올클리어 가능 여부와 전체 제거 계획을 한 번만 계산**하고, 화면 위에 판정(예상 점수)과 지울 사과 박스를 안내합니다.
지우는 것은 직접 드래그로 합니다 (자동 조작 없음).

## 1. 필요한 파일 (모두 같은 폴더에)

| 파일 | 역할 |
|---|---|
| `sum10_pair_finder_v2.py` | 화면 캡처, `REGION` 설정 |
| `apple_detect.py` | 사과 탐지, 숫자 인식, 숫자 모양 학습 |
| `apple_solver.py` | 올클리어 판정 + 전체 제거 계획 계산 (판당 1번) |
| `apple_overlay.py` | 투명 오버레이로 안내 (Windows 전용) |
| `apple_templates.npz` | 학습한 숫자 모양 (학습 후 자동 생성) |
| `install.bat` | Python과 필요한 패키지를 한 번에 설치 (선택) |

## 2. 설치

**가장 쉬운 방법 (Windows 10/11)**: 폴더의 `install.bat`을 더블클릭하세요. Python이 없으면 자동으로 설치하고(이때 창을 닫았다가 `install.bat`을 한 번 더 실행), 필요한 패키지를 설치합니다.

**명령어로 직접** (명령 프롬프트 `cmd`에 한 줄씩):

```
winget install -e --id Python.Python.3.12
```
설치가 끝나면 cmd 창을 닫고 새로 연 뒤:
```
python -m pip install pyautogui opencv-python numpy
```

이미 Python 3.8 이상이 있다면 두 번째 줄만 실행하면 됩니다. `winget`이 없으면 https://www.python.org/downloads/ 에서 설치하되 설치 화면 아래의 **Add python.exe to PATH**를 꼭 체크하세요.

Tesseract(OCR)는 쓰지 않아도 됩니다. 숫자는 `--learn`으로 직접 알려 주는 모양 비교로 읽습니다 (학습 창의 OCR 추천값만 사라집니다).

`sum10_pair_finder_v2.py`가 `import pytesseract`를 하므로, 설치하지 않으려면 그 파일의 아래 부분을 고치세요 (`REGION` 값은 건드리지 않음).

```python
# 기존 (26번째 줄 근처)
import pytesseract
_TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_TESS):
    pytesseract.pytesseract.tesseract_cmd = _TESS

# 변경
try:
    import pytesseract
except ImportError:
    pytesseract = None
```

`--record`(영상 저장)를 쓸 때만 `pip install pillow`가 더 필요합니다 (보통 pyautogui와 함께 이미 설치됩니다).

## 3. 처음 한 번만 (PC가 바뀌면 다시)

```
python sum10_pair_finder_v2.py --calibrate   # 그리드 좌상단/우하단에 마우스 올리고 Enter
```
출력된 `REGION = (...)` 값을 `sum10_pair_finder_v2.py`의 `REGION` 줄에 붙여넣습니다. 그리드 전체를 넉넉히 포함하면 됩니다.

```
python apple_detect.py --check    # 사과가 전부 잡히는지 확인 (apple_check.png 저장)
python apple_detect.py --learn    # 숫자 모양 학습: 키보드 1~9 입력, Space=건너뛰기, Esc=종료
```
`--learn`은 새로 묻는 모양이 없을 때까지 반복하세요.

## 4. 실행

게임을 시작해 **사과가 가득 찬 상태**에서 실행하세요.

```
python apple_overlay.py                # 기본 (분석 시간 6초)
python apple_overlay.py --think 10     # 분석 시간(초). 무작위 판에서 점수를 더 높이고 싶을 때
python apple_overlay.py --record       # 자랑용: 박스·배너가 그려진 영상 저장 (demo.mp4)
python apple_overlay.py --record 내영상.mp4
```

**자랑용 녹화 (`--record`)**: 오버레이 창은 화면 녹화/캡처에 찍히지 않게 설계돼 있습니다. 대신 `--record`를 켜면 프로그램이 게임 화면을 직접 캡처해 박스와 판정 배너를 그려 넣은 mp4를 만듭니다 (깜빡임 없음, 10fps, 게임 영역(REGION)만 저장). ESC로 종료하면 파일이 완성됩니다. 영상에는 마우스 커서와 직접 드래그하는 모습은 담기지 않습니다.

**1) 판정 배너 (화면 위쪽, 계속 표시)** — 점수가 마음에 들지 않으면 게임을 다시 시작하세요. 점수는 사과 1개 = 1점 기준입니다.

| 배너 색 | 의미 |
|---|---|
| 초록 `정답 판 · 올클리어 가능 · 예상 170점` | 모든 사과를 지우는 순서를 찾았습니다 |
| 주황 `무작위 판 의심 · 올클리어 경로 못 찾음 · 예상 N점` | 올클리어 순서를 못 찾음. 계획대로 하면 N점 |
| 빨강 `무작위 판 · 올클리어 불가 (합계 N이 10의 배수 아님)` | 수학적으로 올클리어 불가. 계획대로 하면 예상 점수 |

**2) 박스 안내**
- **올클리어 판**: 정답 순서대로 박스를 **한 번에 1개씩** 보여 줍니다. 지우면 다음 박스가 나옵니다.
- **그 외 판**: 지금 지울 수 있는 계획 단계를 번호 없이 박스로 모두 보여 줍니다 (아무 순서로 지워도 됩니다).
- 계산은 판당 1번뿐이라 지우는 동안 지연이 없습니다. 계획과 다르게 지워도 재계산하지 않고, 아직 유효한 단계를 계속 보여 줍니다.
- 새 판이 시작되면(사과가 다시 늘어나면) 자동으로 다시 분석합니다. 안 되면 `F9`를 누르세요.

| 키 | 동작 |
|---|---|
| `ESC` | 종료 |
| `F9` | 새 판 시작 / 다시 분석 |

**전체 계획을 터미널에서 보기** (오버레이 없이, Mac/Linux 가능)

```
python apple_detect.py --check --solve
```

## 5. 문제 해결

| 증상 | 해결 |
|---|---|
| 사과를 거의 못 찾음 | `REGION`이 그리드를 포함하는지 확인, `apple_detect.py` 상단 `RED_S`/`RED_V` 값을 낮춤 |
| 숫자가 `?`로 많이 나옴 / 틀림 | `python apple_detect.py --learn` 재실행. 잘못 학습했다면 `apple_templates.npz` 삭제 후 다시 학습 |
| 정답 판인데 주황 배너가 나옴 | 숫자 인식이 틀렸을 수 있음. `--check`로 인식 결과(`?`, 잘못 읽은 숫자)를 확인 |
| 박스 위치가 어긋남 | `REGION` 재설정, `F9` 누르기 |
| 오버레이가 깜빡임 | Windows 10 2004 미만에서 나타나는 정상 동작 |
| `tesseract is not installed` 오류 | 학습 완료 후엔 무시 가능. 학습 중이면 Tesseract 설치 (경로 `C:\Program Files\Tesseract-OCR`) |

## 6. 제한 사항

- 오버레이는 **Windows 전용**입니다. 모니터가 여러 대면 게임을 **주 모니터**에 띄우세요.
- 숫자를 한 개라도 잘못 읽으면 계획이 틀릴 수 있습니다. 인식 실패(`?`, 주황 점선) 칸은 계획에서 제외됩니다.
- 판정은 "계획 탐색 결과"입니다. 올클리어 경로를 못 찾았다고 반드시 불가능하다는 뜻은 아닙니다 (합계가 10의 배수가 아닐 때만 확정).
- 게임 규칙상 허용되는 범위에서만 사용하세요.
