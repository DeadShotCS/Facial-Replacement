# Facial-Replacement
Homography based facial replacement using segmentation. This library focuses on utilizing segmentation to create a more pose invariant facial replacement method when using homography.

The basis of this library is made off of reference [2] for dense facial landmark collection.

This is a project for CS 8650, and was turned into a small informal paper.

This was a test project that is not super useful in real applications. Currently using [2] for dense facial landmarks accounts for most of the run time, and using their facial replacment method gives more consistant results. If you can find a way of collecting dense facial landmarks faster this method would be more practical.

# Full Process
<img src="images/transformation_process.JPG" width="600" >

```python
dst_image = 'Path to destination face'
src_image = 'Path to source face'
```

Lines 253-254 are your input images.

# Dense Landmark Collection [2]

https://github.com/YadiraF/PRNet

<img src="images/face_dense.JPG" width="400" >

Dense facial landmarks are used as the foundation for this project so that we can split the face into small segments and project these segments onto the destination face.

# Facial Segmentation Using Landmarks
<img src="images/all_face_segments.JPG" width="300" >

```python
lines_1 = lines_cropped_1
lines_2 = lines_cropped_2
```

Lines 263-264 control partial cropping.

Facial segmentation is just done by creating tons of arrays that hold individual parts of the face. These small parts can be projected onto their corresponding parts on the destination face for more accurate facial replacement. There are some problems accomponied by partial obfuscation due to facial pose. The above code lines attempt to reduce these, but it is a janky way of fixing this problem. Additionally hair and other objects can do this which will result in poor results so discresion must be taken.

# Projection of Face and Facial Replacement
<img src="images/faces_changed.JPG" width="300" >


<img src="images/output2.jpg" width="200" > <img src="images/output3.jpg" width="200" >



# References

[1] G. Bradski. The OpenCV Library. Dr. Dobb’s Journal of Software Tools, 2000. 

[2] Yao Feng, Fan Wu, Xiaohu Shao, Yanfeng Wang, and Xi Zhou. Joint 3d face reconstruction and dense alignment with position map regression network. In ECCV, 2018. (https://github.com/YadiraF/PRNet)

[3] Adam Geitgey. Facial recognition. https://github.com/ageitgey/face_recognition/blob/master/LICENSE, 2017. MIT License. 

[4] Eric Jones, Travis Oliphant, Pearu Peterson, et al. SciPy: Open source scientiﬁc tools for Python. http://www.scipy.org/, 2001.

[5] Travis Oliphant. NumPy: A guide to NumPy. http://www.numpy.org/, 2006. USA: Trelgol Publishing. 

[6] wuhuikai, XiangFugui, and niczem. Faceswap. https://github.com/wuhuikai/FaceSwap, 2018. 
