# Image Recognition Using Squeezenet and Django

Hello, this is a sample image recognition website using django. The web interface is very simple. It mainly allows users to upload a file and the squeezenet will try to predict the top 3 possible predictions.

## Why squeezenet?
It is very lightweight while the performance is comparable to Alexnet.
You can check the paper here: https://arxiv.org/abs/1602.07360

My boss asked me to create a working demo of image recognition hosted on raspberry 1 device. Hence, using anything else other than squeezenet is mostly impossible since Raspberry 1 only have 512MB of RAM.

You can clone the repo and use ./manage.py migrate to run it.


Since I am not a coder at heart (did not take any formal computer-related degree), the codes might be messy. Feel free to contribute and I will be happy for each help.



## Credits
https://github.com/sibtc/simple-file-upload - I am noob at django, hence I cloned this guy's amazing work to make things work!
