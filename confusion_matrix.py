import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from matplotlib.colors import ListedColormap

single_color_cmap = ListedColormap(['#f0f0f0'])
# 2. Confusion Matrix 생성

#cm = confusion_matrix(y_true, y_pred)
cm = np.array([[17989,44],
               [1933,33]])
cm = np.array([[18033,0],
               [1967,0]])
cm = np.array([[11055,6967],
               [567,1411]])
cm = np.array([[11767,6255],
               [660,1318]])
cm_label = np.array([['TN','FP'],
               ['FN','TP']])

fig, ax = plt.subplots(figsize=(5, 5))

# 3. 배경색 직접 칠하기 (대각선: 초록, 나머지: 빨강)
# PPT에서 보기 좋은 파스텔 톤 색상 코드
correct_color = '#A9D08E'  # 연한 초록
wrong_color = '#FF8C8C'    # 연한 빨강

base_color = ['#AAAAAA', '#FF8C8C', '#A9D08E', '#FDDD88']
for i in range(len(cm)):
    for j in range(len(cm)):
        color = base_color[i * 2 + j]
        # 각 칸에 색상 사각형 추가
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor=color, edgecolor='#666666', lw=1.5))
        # 숫자 텍스트 추가
        ax.text(j, i, cm_label[i, j],
        ha="center", va="center",
        color="black", fontsize=30, weight='bold')
# 3. 시각화 설정
# display_labels를 통해 0, 1 대신 실제 의미하는 단어를 넣을 수 있습니다.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=['False', 'True'])

#disp.plot(cmap=single_color_cmap, colorbar=False, ax=ax, text_kw={'color': 'black', 'fontsize': 25})
#disp.plot(cmap=single_color_cmap, colorbar=False, ax=ax)

ax.tick_params(axis='both', which='major', labelsize=15)

# 4. 테두리 추가 (색상이 같으면 칸 구분이 안 될 수 있으므로 검은색 테두리 추가)
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', lw=0.5))

labels = ['False', 'True']
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels, fontsize=15)
ax.set_yticklabels(labels, fontsize=15)
ax.set_xlabel('Predicted Label', fontsize=15)
ax.set_ylabel('Actual Label', fontsize=15)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(1.5, -0.5)
ax.tick_params(left=False, bottom=False)
plt.tight_layout()
plt.show()