# cvman
audio and video tools for all the man to easy use

- A description of your project
- Links to the project's ReadTheDocs page
- A TravisCI button showing the state of the build
- "Quickstart" documentation (how to quickly install and use your project)
- A list of non-Python dependencies (if any) and how to install them

# 打包命令
python setup.py bdist_wheel

# 安装应用
pip3 install cvman-0.0.1-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

# 代码提交
采用双线开发 「master」 「develop」
## develop
  1. git add .
  2. git commit -m "XXX"
  3. git push origin develop

## master
  1. git checkout master
  2. git merge develop
  3. git push origin master
