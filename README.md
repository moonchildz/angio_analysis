# Analyzing angiogenesis on a chip using deep learning-based image processing
Source code for my publication,&lt;Analyzing angiogenesis on a chip using deep learning-based image processing> https://pubs.rsc.org/en/content/articlehtml/2023/lc/d2lc00983h

!! This code implemented the NuSet, https://github.com/yanglf1121/NuSeT For Nuclei segmentation process. I want to show my deepest appreciation to the authors of NuSeT, https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008193


Installation : 
1) Download files or clone this git.
2) At your python terminal, please install all required modules by entering followings:
```
  pip install -r requirements.txt
```

!Important! 

This application is optimized for following versions of modules : 
```
python : 3.7.0
tensorflow-gpu:1.15.0
numpy:1.19.0

```
If error occurs during installation, please refer the git of yanglf1121.

Also, PLEASE MAKE SURE THAT THE IMAGE YOU WANT TO ANALYZE HAS NUCLEI INFORMATION ON BLUE CHANNEL, AND VASCULAR INFORMATION ON GREEN AND RED CHANNEL. 


3) run main_ui.py to see the interface: 
 ![image](https://user-images.githubusercontent.com/66664844/222650817-4a5f24ad-edaa-4e54-a22c-576de61903c7.png)

4) Press'Load Image' button to load image. 
![image](https://user-images.githubusercontent.com/66664844/222651077-f3f40c59-a4cc-4aec-a6ce-bca6137ba0fb.png)

5) Set your threshold value that suits best for your image. 
![image](https://user-images.githubusercontent.com/66664844/222651228-cbef01fb-d2b6-4858-a174-5e6245b349e9.png)
* The preview offers upper-left part of your image. please take note for this. 

6) Press'Analyze Image' button and wait for a while. 

7) If you see 'Fill' window, it means that the process is done.
![image](https://user-images.githubusercontent.com/66664844/222652191-599c69cf-4218-4f2a-bbf7-5ddf107a93e9.png)

8) Check the newly founded csv file and directory. 
![image](https://user-images.githubusercontent.com/66664844/222652374-6b08461d-27b5-4ddb-80c9-8a19aa5679b3.png)
The name of file/dir is decided by the time you run this application.
