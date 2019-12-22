### Docker commands

docker build -t cb:0.0.1 -f Dockerfile .

docker save cb:0.0.1 | gzip > test.tar.gz

docker images

docker rmi [id]

docker image prune

### 주의 사항

yml SUB_DIR에 자기 이름 넣고 건드리지 말기!

upload할 때 Select Flavor p1으로 선택
