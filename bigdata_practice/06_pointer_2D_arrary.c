#include <stdio.h>
void main() {
    static a[3][3] = {{1,2,3},{4,5,6},{7,8,-9}};
    int *pt;
    pt = a[0]; // pt = a 또는 pt = &a[0][0]과 동일
    while(*pt != -9) {
        printf("%d ", *pt);
        pt ++;
    }
}