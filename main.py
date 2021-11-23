import cv2
import numpy as np
# ----- step 1 -----
# 將有宇宙射線的圖片轉為負片
# ----- step 2 -----
# 拉普拉斯邊緣檢測負片
# ----- step 3 -----
# 檢測處理拉普拉斯檢測的負片，找出宇宙射線

# TODO: 優化整個程式流程 變成只要執行一次就好 且過程不存圖(創新檔案)
# 新想法是，每次跑完流程，就調低檢測宇宙射線的判斷閥值，再跑一次流程
# 有機會可以去除乾淨
# TODO: 或是取周圍最亮但mark不為1的點直接取值，或是運算時直接判斷周圍的點若為1就不給權重 也就是動態生成Kernel


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
        if same_color_point_count > 2 or diff_color_point_count > 3:
            return 1
    # 不是宇宙射線
    return 0


def blurred(img, row, col, kernel=np.array([[1, 1, 1],
                                            [1, 0, 0],
                                            [0, 0, 0]])):
    '''
    把指定像素模糊化
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


# TODO: 1=靜態kernel 2=動態生成Kernel 3=改閥值重複流程 4=MODE2+MODE3 5=MODE1+MODE3
MODE = 5
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

        # step 1 存負片
        nag_img = NegativeFilm(img)

        # step 2 拉普拉斯邊緣檢測
        la_img = cv2.Laplacian(nag_img, cv2.CV_16S, ksize=3)
        la_img = cv2.convertScaleAbs(la_img)

        # step 3 檢測處理拉普拉斯檢測的負片，找出宇宙射線
        # 開3x3檢測鏡
        # 判斷中間的灰階值是否幾乎等於白色，且與周遭的值差距很大(like val>210 or something)
        # 是的話就標記他的座標
        for i in range(1, img_rows-1):
            for j in range(1, img_cols-1):
                markedImgMap[i, j] = MarkComsic(la_img, i, j, val=230)
        # step 4 回到普通圖片，從上往下從左到右把那個區塊磨掉
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
        # ----- 成效不佳 -----
        # 用[[2,2,2],[2,0,0],[0,0,0]]的過濾器來處理得到blur1, 對marked2再處理一次得到blur2, blur1比較好
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
        # 如果對blur1 當作原圖 再從step1跑一次流程，可以得到blur1_round2
        SaveImg("./data_set/img_output/"+"newImg.png", img)
    else:
        # 判斷是否為宇宙射線的閥值
        mask_judg = 230
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
            mask_judg -= 10
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
