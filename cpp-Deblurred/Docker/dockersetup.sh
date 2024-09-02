# ubuntu version check
lsb_release -a
# docker install
sudo apt  install docker.io
# no sudo run
sudo groupadd docker
sudo gpasswd -a $USER docker
sudo systemctl restart docker
sudo reboot

