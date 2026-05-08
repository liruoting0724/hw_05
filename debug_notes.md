# Debug 调试记录

## 报错1：matplotlib中文乱码
现象：图表显示方框
原因：无中文字体
修改：设置英文字体DejaVu Sans

## 报错2：文件找不到
现象：No such file or directory
原因：路径含中文或文件名错误
修改：使用纯英文路径

## 报错3：CUDA不可用
现象：CUDA error
修改：自动切换CPU运行

## 报错4：维度不匹配
现象：shape mismatch
修改：保证全连接层输入维度正确
