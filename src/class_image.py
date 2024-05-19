import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
import sknw

class Image:
    
    def __init__(self, path, extension):

        self.path = path
        self.extension = extension
        self.img = cv2.imread(path + '/img.' + extension)
        self.recognized = False
        self.ans = []

    def recognize_circles(self):

        self.recognized = True

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        def sobel(img):
            kernels = np.array([
                                [
                                    [-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]
                                ],
                                [
                                    [1, 2, 1],
                                    [0, 0, 0],
                                    [-1, -2, -1]
                                ],
                               ])

            res = np.zeros_like(img)
            for ker in kernels:
                filtered = cv2.filter2D(src=img, ddepth=-1, kernel=ker)
                res = np.maximum(res, filtered)

            return res

        def kirsch(img):
            kernels = np.array([
                                [
                                    [5, 5, 5],
                                    [-3, 0, -3],
                                    [-3, -3, -3]
                                ],
                                [
                                    [-3, 5, 5],
                                    [-3, 0, 5],
                                    [-3, -3, -3]
                                ],
                                [
                                    [-3, -3, 5],
                                    [-3, 0, 5],
                                    [-3, -3, 5]
                                ],
                                [
                                    [-3, -3, -3],
                                    [-3, 0, 5],
                                    [-3, 5, 5]
                                ],
                                [
                                    [-3, -3, -3],
                                    [-3, 0, -3],
                                    [5, 5, 5]
                                ],
                                [
                                    [-3, -3, -3],
                                    [5, 0, -3],
                                    [5, 5, -3]
                                ],
                                [
                                    [5, -3, -3],
                                    [5, 0, -3],
                                    [5, -3, -3]
                                ],
                                [
                                    [5, 5, -3],
                                    [5, 0, -3],
                                    [-3, -3, -3]
                                ]
                               ])

            res = np.zeros_like(img)
            for ker in kernels:
                filtered = cv2.filter2D(src=img, ddepth=-1, kernel=ker)
                res = np.maximum(res, filtered)

            return res

        def detect_vertices(img):

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            bin_img = cv2.medianBlur(sobel(hsv[:,:,1]) | sobel(hsv[:,:,2]), 9)
            bin_img = np.where(bin_img > 120, 255, 0).astype(np.uint8)
            bin_img = cv2.medianBlur(bin_img, 17)
            element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            bin_img = cv2.dilate(bin_img, element2, iterations=1)

            (numLabels, labels, stats, centroids) = output = cv2.connectedComponentsWithStats(bin_img, 8, cv2.CV_32S)

            return centroids[1:]

        bin_img = cv2.medianBlur(kirsch(hsv[:,:,1]) | kirsch(hsv[:,:,2]), 7)

        bin_img = np.where(bin_img > 120, 255, 0).astype(np.uint8)
        bin_img = cv2.medianBlur(bin_img, 21)

        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, np.ones((14, 14)))

        element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
        bin_img = cv2.erode(bin_img, element2, iterations=1)
        bin_img = cv2.dilate(bin_img, element2, iterations=1)

        skeleton = skeletonize(bin_img)

        graph = sknw.build_sknw(skeleton)

        fig, ax = plt.subplots(1, 3, figsize=(12, 7))

        for i in range(3):
            ax[i].set_axis_off()

        ax[0].imshow(self.img[:,:,[2,1,0]])
        ax[0].set_title('Image')

        ax[1].imshow(skeleton, cmap='gray')
        ax[1].set_title('Skeleton')

        ax[2].imshow(self.img, cmap='gray')

        for (s,e) in graph.edges():
            ps = graph[s][e]['pts']
            ax[2].plot(ps[:,1], ps[:,0], 'green')

        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        ax[2].plot(ps[:,1], ps[:,0], 'r.')
        ver = detect_vertices(self.img)
        ax[2].plot(ver[:, 0], ver[:,1], 'y.')
        ax[2].set_title('Algo results')

        used = [False for j in range(len(nodes))]

        true_v_ind = {i: [] for i in range(len(ver))}
        eps = 30
        for i in range(len(ver)):
            blue_x, blue_y = ver[i][0], ver[i][1]
            for j in range(len(nodes)):
                red_x, red_y = ps[j, 1], ps[j, 0]
                if np.sqrt((red_x - blue_x) ** 2 + (red_y - blue_y) ** 2) < eps:
                    true_v_ind[i].append(j)
                    used[j] = True

        def deg(v):
            cnt = 0
            for (s,e) in graph.edges():
                if s == v or e == v:
                    cnt += 1
            return cnt

        ans = {i: 0 for i in range(1, 6)}
        for i in range(len(ver)):
            if len(true_v_ind[i]) == 0:
                ans[2] += 1
                continue
            new_deg = 0
            for j in range(len(true_v_ind[i])):
                new_deg += deg(true_v_ind[i][j])
            for j in true_v_ind[i]:
                for k in true_v_ind[i]:
                    if (j, k) in graph.edges() or (k, j) in graph.edges():
                        new_deg -= 1
            ans[new_deg] += 1

        for j in range(len(nodes)):
            if not used[j] and deg(j) == 1:
                ans[1] += 1

        title = ''
        for i in range(1, 6):
            title = title + str(ans[i]) + ', '
        title = '(' + title[:-2] + ')'
        fig.suptitle('Topology features: ' + title, fontsize=20)

        plt.tight_layout()
        plt.savefig(self.path + '/process.jpg')

        self.ans = ans
