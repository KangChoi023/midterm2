# 브라우저 점검 도구 (Windows Chrome + Windows node, WSL 에서 실행)

```bash
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu --remote-debugging-port=9333 \
  --user-data-dir='C:\Temp\wikitest' --allow-file-access-from-files about:blank &
N="/mnt/c/Program Files/nodejs/node.exe"; U="file://wsl.localhost/Ubuntu/home/chlrkd023/c_study/midterm2/index.html"
"$N" cdp.js "$U#/c/1/1" "(async()=>document.title)()"          # 식 평가
"$N" shot.js "$U#/c/1/1" "(async()=>1)()" out.png 1360 900      # 실행 후 캡처
"$N" fulltest.js "$U" 1360 900 lessons|code|misc               # 전체 점검
```
끝나면 `--user-data-dir` 에 wikitest 가 들어간 chrome.exe 만 종료한다.
