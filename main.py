import cv2
import numpy as np
# ----- step 1 -----
# 將有宇宙射線的圖片轉為負片
# ----- step 2 -----
# 拉普拉斯邊緣檢測負片
# ----- step 3 -----
# 濾鏡處理拉普拉斯檢測的負片


STEP = 3
# 手動更改路徑
input_path = "./data_set/img_edge/"
output_path = "./data_set/img_edge/"
filename = "data1.png"


def SaveImg(image_name, img):
    cv2.imwrite(image_name, img)


def NegativeFilm(row, col):
    nagativeImg = np.zeros(shape=(row, col, 1), dtype=np.uint8)
    for r in range(row):
        for c in range(col):
            nagativeImg[r, c] = 255 - img[r, c]

    return nagativeImg

# 想避免雜訊的話，目前想法是如果周遭有2或3個鄰居的色差小於一定值或周遭的值直接黑掉，就當宇宙射線
# 如果下方的點也有一些特徵 或許可以在markedMap做不同的標記


def MarkComsic(img, row, col):
    center_value = img[row, col]
    same_color_point_count = 0
    diff_color_point_count = 0
    # 如果中心點是夠白的
    if center_value > 230:
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


if __name__ == '__main__':
    # input灰階圖片
    img = cv2.imread(input_path+filename, 0)
    # 得到圖片大小
    img_rows = img.shape[0]
    img_cols = img.shape[1]
    # 標記宇宙射線位置
    markedImgMap = np.zeros(shape=(img_rows, img_cols, 1), dtype=np.uint8)

    # step 1 存負片
    if STEP == 1:
        SaveImg(output_path+filename, NegativeFilm(img_rows, img_cols))

    # step 2 拉普拉斯邊緣檢測
    elif STEP == 2:
        la_img = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
        la_img = cv2.convertScaleAbs(la_img)
        SaveImg(output_path+filename, la_img)

    # step 3 濾鏡處理拉普拉斯檢測的負片
    # 開3x3檢測鏡
    # 判斷中間的灰階值是否幾乎等於白色，且與周遭的值差距很大(like > 210 or something)
    # 是的話就標記他的座標
    elif STEP == 3:
        la_edge_img = cv2.imread("./data_set/img_edge/"+filename, 0)
        counter = 0
        for i in range(1, img_rows-1):
            for j in range(1, img_cols-1):
                markedImgMap[i, j] = MarkComsic(la_edge_img, i, j)
        # SaveImg("./data_set/img_output/"+filename, markedImgMap)
    # step 4 回到普通圖片，從上往下從左到右把那個區塊用內插法磨掉
    elif STEP == 4:
        # cv2.filter2D(img, -1, kernel)
        print("回到普通圖片，從上往下從左到右把那個區塊用內插法磨掉")
    # # 二值化
    # ret, th1 = cv2.threshold(la_img, 127, 255, cv2.THRESH_BINARY)

    # canny 太銳利 看不到天體或星系星雲了
    # canny_img = cv2.Canny(img, 50, 110)

    # cv2.imshow("nor", img)
    # cv2.imshow("naga", NegativeFilm(img_rows, img_cols))
    # cv2.waitKey()
    # cv2.destroyAllWindows()
