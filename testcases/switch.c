#include <stdio.h>

int main() {
    int x = 10;
    switch (x) {
        case 1:
            printf("one\n");
            break;
        case 2:
            printf("two\n");
            break;
        default:
            printf("other\n");
            break;
    }
    return 0;
}