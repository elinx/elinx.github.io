---
layout: single
title: IR Translation
date: 2021-06-30 21:51 +0800
categories: compiler design
classes: wide
---
1. 为什么(LLVM)IR里边没有logical and `&&` 和 logical or `||` 指令?

    似乎没有必要，当做整数操作就行了，不需要一条单独的指令。

2. 那么在做AST转IR的时候碰到logical and(`&&`)和logical or(`||`)表达式的时候要怎么处理呢?

    可以将一个语句转换成嵌套的条件判断语句，比如一个单纯的`a && b`语句就可以先分配一个临时变量，然后判断左边的操作数`a`，如果为`true`就把右边的操作数`b`转成bool值就是表达式的最终结果，类似下面：

    ```c++
    bool tmp;
    if (a) {
        tmp = (bool)b;
    } else {
        tmp = false;
    }
    ```

    相似的对于表达式`a || b`，可以转换为：

    ```c++
    bool tmp;
    if (a) {
        tmp = true;
    } else {
        tmp = (bool)b;
    }
    ```

3. 假若`if`语句没有`else`语句应该怎么安排basic blocks?

    补一个空的block块，这个块只有一个`BlockExit`没有其他内容，当然这个块儿也就成了死块儿，永远不会有节点到达。