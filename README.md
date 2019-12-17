# 2019_DL_Project

Class: Neural Network And Deep Learning (Graduate Class, Dept. of Data Science, SNUT) \
Professor: Sangheum Hwang \
Semester: Fall 

## Team Members

| Students Number | Name |
| :------: | :------: |
| 18512083 | Jihyo Kim | 
| 19510099 | Hongjea Park | 
| 19510101 | Geongyu Lee | 
| 19512024 | Sohee Ha |

## Data Sets Statistics 

| Classes | Images | Objects |
| -------- | :------: | --------- |
| Aeroplane | 116 | 136 |
| Bicycle | 97 | 124 |
| Bird | 136 | 176 |
| Boat | 106 | 153 |
| Bottle | 117 | 176 |
| Bus | 99 | 147 |
| Car | 161 | 276 |
| Cat | 169 | 195 |
| Chair | 170 | 348 |
| Cow | 82 | 161 |
| Diningtable | 104 | 108 |
| Dog | 156 | 194 |
| Horse | 105 | 136 |
| Motorbike | 103 | 126 |
| Person | 526 | 925 |
| Pottedplant | 115 | 209 |
| Sheep | 83 | 209 |
| Sofa | 117 | 138 |
| Train | 106 | 118 |
| Tvmonitor | 113 | 148 |
| Total | 1928 | 4203 |

## Images 

![KakaoTalk_20191215_222535812](https://user-images.githubusercontent.com/37532168/70934552-82781800-2081-11ea-919a-f4cb68f4bd20.jpg)
![KakaoTalk_20191215_222408152](https://user-images.githubusercontent.com/37532168/70934553-82781800-2081-11ea-8958-556c1363ecc9.png)


![KakaoTalk_20191215_223100149](https://user-images.githubusercontent.com/37532168/70934576-8c9a1680-2081-11ea-8de0-616d62fc5bf7.jpg)
![KakaoTalk_20191215_223033640](https://user-images.githubusercontent.com/37532168/70934582-8d32ad00-2081-11ea-9a81-1de3980ae7c3.png)


![KakaoTalk_20191215_222649744](https://user-images.githubusercontent.com/37532168/70934577-8c9a1680-2081-11ea-8de1-ce9ef3bc90b4.jpg)
![KakaoTalk_20191215_222700786](https://user-images.githubusercontent.com/37532168/70934578-8c9a1680-2081-11ea-8ec1-7ba6de149dad.png)


![KakaoTalk_20191215_222745577](https://user-images.githubusercontent.com/37532168/70934581-8d32ad00-2081-11ea-9171-6c8ccf99c1d8.jpg)
![KakaoTalk_20191215_222714869](https://user-images.githubusercontent.com/37532168/70934579-8c9a1680-2081-11ea-91b3-17e1e07b3867.png)


## Major Tasks

| 	 | Base | Adversarial |
| :------: | :------: | :------: |
| baseline |  		 |   		 | 
| label smoothing | 		 |   		 | 
| cut-out | 		 |   		 | 
| label smoothing + cut-out | 		 |  		 | 


## Segmentation Results

[DOG]
![result_50_eopch_Segmentation_Smoothing_0](https://user-images.githubusercontent.com/37532168/70983492-30280d00-20fc-11ea-8a1a-26460f89b2ab.png)
![result_50_eopch_Segmentation Baseline_0](https://user-images.githubusercontent.com/37532168/70983494-30280d00-20fc-11ea-9bee-588824d72bc3.png)
![result_50_eopch_Segmentation_ADV_0](https://user-images.githubusercontent.com/37532168/70983495-30c0a380-20fc-11ea-8ed1-7deb2fb96e46.png)
![result_50_eopch_Segmentation_ADV_CutOut_0](https://user-images.githubusercontent.com/37532168/70983497-30c0a380-20fc-11ea-87dd-079f05ce3bdb.png)
![result_50_eopch_Segmentation_ADV_Smooth_0](https://user-images.githubusercontent.com/37532168/70983498-30c0a380-20fc-11ea-891e-33afc0913fc2.png)
![result_50_eopch_Segmentation_CutOut_0](https://user-images.githubusercontent.com/37532168/70983499-30c0a380-20fc-11ea-8457-98b36b6635f9.png)

[HUMAN]
![result_50_eopch_Segmentation_Smoothing_6](https://user-images.githubusercontent.com/37532168/70983645-6ebdc780-20fc-11ea-8ec5-df42c6715e58.png)
![result_50_eopch_Segmentation Baseline_6](https://user-images.githubusercontent.com/37532168/70983647-6ebdc780-20fc-11ea-93be-09f9b37cfb29.png)
![result_50_eopch_Segmentation_ADV_6](https://user-images.githubusercontent.com/37532168/70983651-6f565e00-20fc-11ea-968b-86b89d7fb07d.png)
![result_50_eopch_Segmentation_ADV_CutOut_6](https://user-images.githubusercontent.com/37532168/70983652-6f565e00-20fc-11ea-990c-7421ff32b8d0.png)
![result_50_eopch_Segmentation_ADV_Smooth_6](https://user-images.githubusercontent.com/37532168/70983653-6f565e00-20fc-11ea-86ae-ba3380912ced.png)
![result_50_eopch_Segmentation_CutOut_6](https://user-images.githubusercontent.com/37532168/70983654-6feef480-20fc-11ea-83bb-142551ca3d25.png)


## Future Work 

- Pre-Trainied 모델로 학습
- Adversarial Training을 통하여 얻는 이득에 비하여 얼마나 이득이 있었는지 계산하기
- Performence measure 추가 하기

## Pre-Trained Model Download 

[Download Here](https://drive.google.com/drive/folders/18dD_dwOiHBItlqZpaBVh9C0YLx2B6mwO?usp=sharing)




## Privacy & Term 

@misc{pascal-voc-2010,
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2010 {(VOC2010)} {R}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2010/workshop/index.html"}
	
	
## References

[1] Goibert, M., & Dohmatob, E. (2019). Adversarial Robustness via Adversarial Label-Smoothing. arXiv preprint arXiv:1906.11567. \
[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778). \
[3] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham. \
[4] Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572. \
[5] Müller, R., Kornblith, S., & Hinton, G. (2019). When Does Label Smoothing Help?. arXiv preprint arXiv:1906.02629. \
[6] DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552.  





