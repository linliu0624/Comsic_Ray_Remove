import cv2
import numpy as np
# ----- step 1 -----
# 拉普拉斯邊緣檢測影像
# ----- step 2 -----
# 檢測處理拉普拉斯檢測的負片，找出宇宙射線
# ----- step 3 -----
# 回到普通圖片，從上往下從左到右把那個區塊磨掉

# 想調整模式可以調整MODE變數的初始值
# 想要使用最小濾波器，可以調整comment out第148行，使用149行，並將MODE設定為1

def SaveImg(image_name, img):
    cv2.imwrite(image_name, img)


def NegativeFilm(img):
    img_rows = img.shape[0]
    img_cols = img.shape[1]
    nagativeImg = np.zeros(shape=(img_rows, img_cols, 1), dtype=np.uint8)
    for r in range(img_rows):
        for c in range(img_cols):
            nagativeImg[r, c] = 255 - img[r, c]

    return nagativeImg


def MarkComsic(img, row, col, val=230):
    '''
    想避免雜訊的話，目前想法是如果周遭有2或3個鄰居的色差小於一定值或周遭的值直接黑掉，就當宇宙射線。 \n
    如果下方的點也有一些特徵 或許可以在markedMap做不同的標記。

    '''
    center_value = img[row, col]
    same_color_point_count = 0
    diff_color_point_count = 0
    # 如果中心點是夠白的
    if center_value > val:
        # 判定周遭有多少顏色類似的點, 且是否有落差極大的點
        for i in range(-1, 2):
            for j in range(-1, 2):
                # 正中間的點不計算
                if i == 0 and j == 0:
                    continue
                if img[row+i, col+j] > center_value-5 or img[row+i, col+j] < center_value + 5:
                    same_color_point_count += 1
                elif img[row+i, col+j] < center_value/2:
                    diff_color_point_count += 1
        # 是宇宙射線
        if same_color_point_count > 2:
            return 1
    # 不是宇宙射線
    return 0


def laplace(img, ksize=3):
    '''
    自製拉普拉斯邊緣檢測
    '''
    height = img.shape[0]
    width = img.shape[1]
    output_image = np.zeros(shape=(height, width))
    if ksize == 3:
        filter = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])

    for i in range(1, height-1):
        for j in range(1, width-1):
            output_image[i, j] = abs(np.sum(img[i-1:i+2, j-1:j+2]*filter))
    return np.uint8(output_image)


def blurred(img, row, col, kernel=np.array([[1, 1, 1],
                                            [1, 0, 0],
                                            [0, 0, 0]])):
    '''
    把指定像素做相鄰像素平均法
    '''
    # kernel = np.array([[2, 2, 2],
    #                    [2, 0, 1],
    #                    [1, 1, 1]])

    value = np.array([[img[row-1, col-1], img[row-1, col], img[row-1, col+1]],
                      [img[row, col-1], img[row, col], img[row, col+1]],
                      [img[row+1, col-1], img[row+1, col], img[row+1, col+1]]])
    sum = np.sum(kernel)
    if sum == 0:
        return 125
    return np.sum(np.floor(kernel/sum * value))


def min_filtering(img, row, col):
    '''
    最小濾波器
    value為原始圖片中指定像素與其8鄰點
    '''
    value = np.array([[img[row-1, col-1], img[row-1, col], img[row-1, col+1]],
                      [img[row, col-1], img[row, col], img[row, col+1]],
                      [img[row+1, col-1], img[row+1, col], img[row+1, col+1]]])

    return np.min(value)


# 1=靜態kernel 2=動態生成Kernel 3=改閥值重複流程 4=MODE2+MODE3
MODE = 4
input_path = "./data_set/img_with_cosmic/"
output_path = "./data_set/img_output/"
filename = "data1.png"

if __name__ == '__main__':
    if MODE < 3:
        # input灰階圖片
        img = cv2.imread(input_path+filename, 0)
        # 得到圖片大小
        img_rows = img.shape[0]
        img_cols = img.shape[1]
        # 標記宇宙射線位置
        markedImgMap = np.zeros(shape=(img_rows, img_cols, 1), dtype=np.uint8)

        # step 1 拉普拉斯邊緣檢測
        la_img = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
        la_img = cv2.convertScaleAbs(la_img)

        # step 2 檢測處理拉普拉斯檢測的負片，找出宇宙射線
        # 開3x3遮罩
        # 判斷中間的灰階值是否幾乎等於白色，且與周遭的值差距很大(like val>210 or something)
        # 是的話就標記他的座標
        markedImgMap = np.zeros(shape=(img_rows, img_cols, 1), dtype=np.uint8)
        for i in range(1, img_rows-1):
            for j in range(1, img_cols-1):
                markedImgMap[i, j] = MarkComsic(la_img, i, j, val=150)
        # step 3 回到普通圖片，從上往下從左到右把那個區塊磨掉
        for i in range(1, img_rows-1):
            for j in range(1, img_cols-1):
                if markedImgMap[i, j] == 1:
                    if MODE == 2:
                        kernel = np.zeros(shape=(3, 3))
                        for x in range(-1, 2):
                            for y in range(-1, 2):
                                if markedImgMap[i+x, j+y] == 1:
                                    kernel[x+1, y+1] = 0
                                else:
                                    kernel[x+1, y+1] = 1
                        img[i, j] = blurred(img, i, j, kernel)
                        markedImgMap[i, j] = 0
                    else:
                        img[i, j] = blurred(img, i, j)
                        # img[i, j] = min_filtering(img, i, j)
        # ----- 成效不佳 -----
        #             # 之後再對2的部分做一是磨平
        #             if markedImgMap[i-1, j] != 1:
        #                 markedImgMap[i-1, j] = 2
        #             if markedImgMap[i, j-1] != 1:
        #                 markedImgMap[i, j-1] = 2
        #             if markedImgMap[i+1, j] != 1:
        #                 markedImgMap[i+1, j] = 2
        #             if markedImgMap[i, j+1] != 1:
        #                 markedImgMap[i, j+1] = 2

        # for i in range(1, img_rows-1):
        #     for j in range(1, img_cols-1):
        #         if markedImgMap[i, j] == 2:
        #             origin_image[i, j] = blurred(origin_image, i, j)
        # --------------------
        SaveImg("./data_set/img_output/"+"newImg.png", img)
    else:
        # 判斷是否為宇宙射線的閥值
        mask_judg = 180
        for n in range(2):
            # input灰階圖片
            if n == 0:
                img = cv2.imread(input_path+filename, 0)
            else:
                img = cv2.imread("./data_set/temp.png", 0)
            # 得到圖片大小
            img_rows = img.shape[0]
            img_cols = img.shape[1]
            # 標記宇宙射線位置
            markedImgMap = np.zeros(
                shape=(img_rows, img_cols, 1), dtype=np.uint8)

            # step 1 存負片
            nag_img = NegativeFilm(img)

            # step 2 拉普拉斯邊緣檢測
            la_img = cv2.Laplacian(nag_img, cv2.CV_16S, ksize=3)
            la_img = cv2.convertScaleAbs(la_img)

            # step 3 檢測處理拉普拉斯檢測的負片，找出宇宙射線
            for i in range(1, img_rows-1):
                for j in range(1, img_cols-1):
                    markedImgMap[i, j] = MarkComsic(
                        la_img, i, j, val=mask_judg)
            mask_judg -= 40
            # step 4 回到普通圖片，從上往下從左到右把那個區塊磨掉
            for i in range(1, img_rows-1):
                for j in range(1, img_cols-1):
                    if markedImgMap[i, j] == 1:
                        if MODE == 4:
                            kernel = np.zeros(shape=(3, 3))
                            for x in range(-1, 2):
                                for y in range(-1, 2):
                                    if markedImgMap[i+x, j+y] == 1:
                                        kernel[x+1, y+1] = 0
                                    else:
                                        kernel[x+1, y+1] = 1
                            img[i, j] = blurred(img, i, j, kernel)
                            markedImgMap[i, j] = 0
                        else:
                            img[i, j] = blurred(img, i, j)
            SaveImg("./data_set/temp.png", img)

        SaveImg("./data_set/img_output/"+"newImg.png", img)

    # Sobel算子測試
    # img = cv2.imread(input_path+filename, 0)
    # sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    # sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    # sobelX = np.uint8(np.absolute(sobelX))
    # sobelY = np.uint8(np.absolute(sobelY))
    # sobelCombined = cv2.bitwise_or(sobelX, sobelY)
    # SaveImg("./data_set/img_edge/"+"sobelx_data1.png", sobelX)
    # SaveImg("./data_set/img_edge/"+"sobely_data1.png", sobelY)
    # SaveImg("./data_set/img_edge/"+"sobelxy_data1.png", sobelCombined)
