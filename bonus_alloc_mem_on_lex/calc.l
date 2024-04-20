%{
    #include <string>
    #define YYSTYPE std::string *
    #include "calc.tab.h"
%}
%option yylineno
%%
[a-zA-Z]+ {
    yylloc.last_line=yylloc.first_line =yylineno;
    yylval = new std::string[1]{yytext};
    return STRING;
}
\n {
    yylloc.last_line=yylloc.first_line =yylineno;
    return *yytext;
}
[ \t] ; /* skip whitespace */
. {
    yylloc.last_line=yylloc.first_line =yylineno;
    return *yytext;
}
%%
int yywrap(void) {
    return 1;
}
