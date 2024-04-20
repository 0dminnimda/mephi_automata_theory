%{
    #include <string>
    #include <iostream>
    #define YYSTYPE std::string *
    int yylex(void);
    void yyerror(const char *);
    extern FILE * yyin;
%}
%token STRING
%left '+'
%%
program:
    program '\n'
    | program expr '\n' { std::cout << (*$2) << std::endl; delete[] $2; }
    | program error '\n' { std::cout << "some error at line " << @2.first_line << std::endl; delete[] $2; yyerrok; }
    |
    ;
expr:
    STRING { $$ = $1; }
    | expr '+' expr { $$ = new std::string[1]{*$1 + *$3}; delete[] $1; delete[] $3; }
    /*| expr '-' expr { $$ = $1 - $3; }*/
    | '(' expr ')' { $$ = $2; }
    ;
%%
void yyerror(const char *s) {
    fprintf(stderr, "%s\n", s);
}
int main(int argc, const char* argv[]) {
    yyin = fopen (argv[1], "r");
    if (!yyin)
        yyin = stdin;
    yyparse();
    std::cout << "Total number of strings with errors: " << yynerrs << std::endl;
    return 0;
}
