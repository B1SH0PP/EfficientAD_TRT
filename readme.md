# v240421-1615

* 模型文件提取码:

  链接：https://pan.baidu.com/s/1Tt4gXlIADZosZGCgX2vd6g?pwd=66eo 
  提取码：66eo

  下载`models.zip`文件后, 将模型文件放入`models`文件夹下:

  ```
  datas
  |-- images
  |-- models
  ```

  

* 删除了一些`yolov5`修改而来中冗余的代码;

# v240418-1615

* 基于tensorrtx下的yolov5的修改而来;

  目前代码和文件有些冗余, 一些多余无用的文件没有还没删除赶紧;

* 总体结构已经完成, 已经可以正常导出异常图 (8bit) 和热力图;



# torch-data

input:

![image-20240416004710921](readme.assets/image-20240416004710921.png)



ImageNetNormal:

![image-20240416004651392](readme.assets/image-20240416004651392.png)



teacher:

![image-20240416010100044](readme.assets/image-20240416010100044.png)

![image-20240416010238646](readme.assets/image-20240416010238646.png)

![image-20240416024621713](readme.assets/image-20240416024621713.png)



teacher_output_normal:

![image-20240416024802004](readme.assets/image-20240416024802004.png)



student:

![image-20240416145400814](readme.assets/image-20240416145400814.png)

![image-20240416145511808](readme.assets/image-20240416145511808.png)

![image-20240416145607924](readme.assets/image-20240416145607924.png)

![image-20240416104637908](readme.assets/image-20240416104637908.png)



distance_student&teacher:

![image-20240416150005406](readme.assets/image-20240416150005406.png)



AE:

![image-20240416182429720](readme.assets/image-20240416182429720.png)



distance_st:

![image-20240416215753267](readme.assets/image-20240416215753267.png)



distance_stae:

