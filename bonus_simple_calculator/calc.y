%{
 #include <stdio.h>
 #include <math.h>
 int yylex(void);
 void yyerror(const char *);
%}

%token INTEGER

%%
program:
 program '\n' { printf("empty string\n", $2); }
 | program expr '\n' { printf("result = %d\n", $2); }
 | { printf("enter a expression:\n"); }
 ;

expr:
 expr '+' term { $$ = $1 + $3; printf("%d + %d = %d\n", $1, $3, $$); }
 | expr '-' term { $$ = $1 - $3; printf("%d - %d = %d\n", $1, $3, $$); }
 | term { $$ = $1; }
 ;

term:
 term '*' power { $$ = $1 * $3; printf("%d * %d = %d\n", $1, $3, $$); }
 | term '/' power { $$ = $1 / $3; printf("%d / %d = %d\n", $1, $3, $$); }
 | power { $$ = $1; }
 ;

power:
 factor '^' power { $$ = pow($1, $3); printf("%d ^ %d = %d\n", $1, $3, $$); }
 | factor { $$ = $1; }
 ;

factor:
 INTEGER { printf("int = %d\n", $1, $$); }
 | '(' expr ')' { $$ = $2; printf("( %d ) = %d\n", $2, $$); }
 ;
%%

void yyerror(const char *s) {
 fprintf(stderr, "%s\n", s);
}

int yylex(void)
{
 int c;
 while ((c = getchar()) == ' '|| c == '\t')
 ;
 if (isdigit(c))
 {
 ungetc (c, stdin);
 scanf ("%d", &yylval);
 return INTEGER;
 }
 if (c == EOF)
 return 0;
 return c;
} 

int main(void) {
 yyparse();
 return 0;
}