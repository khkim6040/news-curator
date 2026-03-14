#!/bin/bash
# news-curator launchd 설치 스크립트
# 사용법: bash install_launchd.sh

PLIST_NAME="com.gwanho.news-curator.plist"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${REPO_DIR}/${PLIST_NAME}"
DEST="$HOME/Library/LaunchAgents/${PLIST_NAME}"

PYTHON_PATH="$(which python3)"
if [ -z "$PYTHON_PATH" ]; then
    echo "✗ python3을 찾을 수 없습니다. PATH를 확인해 주세요."
    exit 1
fi

# 기존 작업 언로드 (이미 등록된 경우)
launchctl bootout "gui/$(id -u)/${PLIST_NAME%.plist}" 2>/dev/null

# 템플릿 치환 후 plist 설치
sed -e "s|__REPO_DIR__|${REPO_DIR}|g" \
    -e "s|__PYTHON_PATH__|${PYTHON_PATH}|g" \
    -e "s|__HOME_DIR__|${HOME}|g" \
    "$SRC" > "$DEST"
echo "✓ plist 설치 완료: $DEST"
echo "  Python: $PYTHON_PATH"
echo "  Repo:   $REPO_DIR"

# 등록
launchctl bootstrap "gui/$(id -u)" "$DEST"
echo "✓ launchd 등록 완료"

# 상태 확인
launchctl print "gui/$(id -u)/com.gwanho.news-curator" 2>/dev/null | head -5
echo ""
echo "매일 오전 8시에 news_curator.py가 실행됩니다."
echo ""
echo "유용한 명령어:"
echo "  상태 확인: launchctl print gui/\$(id -u)/com.gwanho.news-curator"
echo "  즉시 실행: launchctl kickstart gui/\$(id -u)/com.gwanho.news-curator"
echo "  제거:      launchctl bootout gui/\$(id -u)/com.gwanho.news-curator"
