.PHONY: com
com:
	flex -L calc.l
	bison -dl calc.y
	g++ calc.tab.c lex.yy.c -o calc.out -lm -g  # -fsanitize=address 

.PHONY: run
run:
	valgrind --leak-check=full ./calc.out

.PHONY: clean
clean:
	rm ./*.c
	rm ./*.h
	rm ./*.out
