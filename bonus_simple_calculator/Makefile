.PHONY: com
com:
	flex calc.l
	bison -dy calc.y
	gcc y.tab.c -o calc.out -lm

.PHONY: clean
clean:
	rm ./*.c
	rm ./*.h
	rm ./*.out
