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

