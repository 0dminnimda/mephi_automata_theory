%{
 #include "y.tab.h"
 #include <stdlib.h>
 void yyerror(const char *);
%}
%%
[0-9]+ {
 yylval = atoi(yytext);
 return INTEGER;
 }
[-+*/()\n] { return *yytext; }
[ \t] ; /* skip whitespace */
. yyerror("Unknown character");
%%
int yywrap(void) {
 return 1;
}
