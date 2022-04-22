#include <stdio.h>
void main() {
    static int score[4][3] = {{90, 90, 90},{80, 80, 80}, {70,70,70}, {60, 60, 60}};
    // 2차원 배열의 선언과 초기화 //
    int sum, i, j;
    printf("번호 국어 수학 영어 합계\n");
    for(i=0; i<4; ++i) {
        sum = 0;
        printf("%-7d", i+1);
        for(j=0; j<3; j++) {
            printf("%-7d", score[i][j]);
            sum += score[i][j];
        }
        printf("%-7d\n", sum);
    }
}