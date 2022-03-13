# mlasterchef

# Requirements
Last login: Tue Feb  8 21:53:44 on ttys001

The default interactive shell is now zsh.
To update your account to use zsh, please run `chsh -s /bin/zsh`.
For more details, please visit https://support.apple.com/kb/HT208050.
MacBook-Air:~ joshuacastle$ source django-env/bin/activate
(django-env) MacBook-Air:~ joshuacastle$ pip install matplotlib
Collecting matplotlib
  Downloading matplotlib-3.5.1-cp38-cp38-macosx_10_9_x86_64.whl (7.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.3/7.3 MB 11.8 MB/s eta 0:00:00
Requirement already satisfied: numpy>=1.17 in ./django-env/lib/python3.8/site-packages (from matplotlib) (1.22.3)
Collecting cycler>=0.10
  Downloading cycler-0.11.0-py3-none-any.whl (6.4 kB)
Requirement already satisfied: python-dateutil>=2.7 in ./django-env/lib/python3.8/site-packages (from matplotlib) (2.8.2)
Collecting pillow>=6.2.0
  Downloading Pillow-9.0.1-cp38-cp38-macosx_10_10_x86_64.whl (3.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.0/3.0 MB 12.1 MB/s eta 0:00:00
Collecting packaging>=20.0
  Downloading packaging-21.3-py3-none-any.whl (40 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.8/40.8 KB 961.5 kB/s eta 0:00:00
Collecting fonttools>=4.22.0
  Downloading fonttools-4.30.0-py3-none-any.whl (898 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 898.1/898.1 KB 9.2 MB/s eta 0:00:00
Collecting pyparsing>=2.2.1
  Downloading pyparsing-3.0.7-py3-none-any.whl (98 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98.0/98.0 KB 2.6 MB/s eta 0:00:00
Collecting kiwisolver>=1.0.1
  Downloading kiwisolver-1.3.2-cp38-cp38-macosx_10_9_x86_64.whl (61 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.6/61.6 KB 1.6 MB/s eta 0:00:00
Requirement already satisfied: six>=1.5 in ./django-env/lib/python3.8/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
Installing collected packages: pyparsing, pillow, kiwisolver, fonttools, cycler, packaging, matplotlib
Successfully installed cycler-0.11.0 fonttools-4.30.0 kiwisolver-1.3.2 matplotlib-3.5.1 packaging-21.3 pillow-9.0.1 pyparsing-3.0.7
(django-env) MacBook-Air:~ joshuacastle$ pip install sklearn
Collecting sklearn
  Downloading sklearn-0.0.tar.gz (1.1 kB)
  Preparing metadata (setup.py) ... done
Collecting scikit-learn
  Downloading scikit_learn-1.0.2-cp38-cp38-macosx_10_13_x86_64.whl (7.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.9/7.9 MB 10.9 MB/s eta 0:00:00
Collecting scipy>=1.1.0
  Downloading scipy-1.8.0-cp38-cp38-macosx_12_0_universal2.macosx_10_9_x86_64.whl (55.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 55.3/55.3 MB 11.3 MB/s eta 0:00:00
Requirement already satisfied: numpy>=1.14.6 in ./django-env/lib/python3.8/site-packages (from scikit-learn->sklearn) (1.22.3)
Collecting joblib>=0.11
  Downloading joblib-1.1.0-py2.py3-none-any.whl (306 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 307.0/307.0 KB 5.8 MB/s eta 0:00:00
Collecting threadpoolctl>=2.0.0
  Downloading threadpoolctl-3.1.0-py3-none-any.whl (14 kB)
Using legacy 'setup.py install' for sklearn, since package 'wheel' is not installed.
Installing collected packages: threadpoolctl, scipy, joblib, scikit-learn, sklearn
  Running setup.py install for sklearn ... done
Successfully installed joblib-1.1.0 scikit-learn-1.0.2 scipy-1.8.0 sklearn-0.0 threadpoolctl-3.1.0
(django-env) MacBook-Air:~ joshuacastle$ pip install transformers
Collecting transformers
  Downloading transformers-4.17.0-py3-none-any.whl (3.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.8/3.8 MB 3.3 MB/s eta 0:00:00
Collecting pyyaml>=5.1
  Downloading PyYAML-6.0-cp38-cp38-macosx_10_9_x86_64.whl (192 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 192.2/192.2 KB 2.6 MB/s eta 0:00:00
Collecting huggingface-hub<1.0,>=0.1.0
  Downloading huggingface_hub-0.4.0-py3-none-any.whl (67 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 67.0/67.0 KB 1.8 MB/s eta 0:00:00
Collecting filelock
  Downloading filelock-3.6.0-py3-none-any.whl (10.0 kB)
Requirement already satisfied: requests in ./django-env/lib/python3.8/site-packages (from transformers) (2.27.1)
Collecting sacremoses
  Downloading sacremoses-0.0.47-py2.py3-none-any.whl (895 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 895.2/895.2 KB 4.4 MB/s eta 0:00:00
Requirement already satisfied: numpy>=1.17 in ./django-env/lib/python3.8/site-packages (from transformers) (1.22.3)
Collecting regex!=2019.12.17
  Downloading regex-2022.3.2-cp38-cp38-macosx_10_9_x86_64.whl (289 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 289.1/289.1 KB 2.7 MB/s eta 0:00:00
Requirement already satisfied: packaging>=20.0 in ./django-env/lib/python3.8/site-packages (from transformers) (21.3)
Collecting tokenizers!=0.11.3,>=0.11.1
  Downloading tokenizers-0.11.6-cp38-cp38-macosx_10_11_x86_64.whl (3.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.6/3.6 MB 4.2 MB/s eta 0:00:00
Collecting tqdm>=4.27
  Downloading tqdm-4.63.0-py2.py3-none-any.whl (76 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 76.6/76.6 KB 2.0 MB/s eta 0:00:00
Collecting typing-extensions>=3.7.4.3
  Downloading typing_extensions-4.1.1-py3-none-any.whl (26 kB)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in ./django-env/lib/python3.8/site-packages (from packaging>=20.0->transformers) (3.0.7)
Requirement already satisfied: certifi>=2017.4.17 in ./django-env/lib/python3.8/site-packages (from requests->transformers) (2021.10.8)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in ./django-env/lib/python3.8/site-packages (from requests->transformers) (1.26.8)
Requirement already satisfied: charset-normalizer~=2.0.0 in ./django-env/lib/python3.8/site-packages (from requests->transformers) (2.0.10)
Requirement already satisfied: idna<4,>=2.5 in ./django-env/lib/python3.8/site-packages (from requests->transformers) (3.3)
Collecting click
  Downloading click-8.0.4-py3-none-any.whl (97 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 97.5/97.5 KB 2.0 MB/s eta 0:00:00
Requirement already satisfied: joblib in ./django-env/lib/python3.8/site-packages (from sacremoses->transformers) (1.1.0)
Requirement already satisfied: six in ./django-env/lib/python3.8/site-packages (from sacremoses->transformers) (1.16.0)
Installing collected packages: tokenizers, typing-extensions, tqdm, regex, pyyaml, filelock, click, sacremoses, huggingface-hub, transformers
Successfully installed click-8.0.4 filelock-3.6.0 huggingface-hub-0.4.0 pyyaml-6.0 regex-2022.3.2 sacremoses-0.0.47 tokenizers-0.11.6 tqdm-4.63.0 transformers-4.17.0 typing-extensions-4.1.1
(django-env) MacBook-Air:~ joshuacastle$ pip install torch
Collecting torch
  Downloading torch-1.11.0-cp38-none-macosx_10_9_x86_64.whl (129.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 129.9/129.9 MB 9.2 MB/s eta 0:00:00
Requirement already satisfied: typing-extensions in ./django-env/lib/python3.8/site-packages (from torch) (4.1.1)
Installing collected packages: torch
Successfully installed torch-1.11.0
(django-env) MacBook-Air:~ joshuacastle$ pip install nltk
Collecting nltk
  Downloading nltk-3.7-py3-none-any.whl (1.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 2.7 MB/s eta 0:00:00
Requirement already satisfied: regex>=2021.8.3 in ./django-env/lib/python3.8/site-packages (from nltk) (2022.3.2)
Requirement already satisfied: click in ./django-env/lib/python3.8/site-packages (from nltk) (8.0.4)
Requirement already satisfied: joblib in ./django-env/lib/python3.8/site-packages (from nltk) (1.1.0)
Requirement already satisfied: tqdm in ./django-env/lib/python3.8/site-packages (from nltk) (4.63.0)
Installing collected packages: nltk
Successfully installed nltk-3.7
(django-env) MacBook-Air:~ joshuacastle$ 
