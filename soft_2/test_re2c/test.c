// re2c test.c -o test.out.c -i --case-ranges --dump-nfa --dump-dfa-raw --dump-dfa-tree --dump-dfa-det --dump-closure-stats --no-optimize-tags -T --emit-dot > out.dot 2>&1
// --posix-captures 
// /c/Tools/re2c/.build/re2c.exe test.c --case-ranges --dump-dfa-raw --dump-dfa-det --dump-closure-stats --dump-dfa-min --dump-nfa --no-optimize-tags  -T --emit-dot > out.dot 2>&1
// /c/Tools/re2c/.build/re2c.exe test.c --case-ranges --dump-dfa-raw --dump-dfa-det --dump-closure-stats --dump-dfa-min --dump-nfa --no-optimize-tags --leftmost-captures -T --emit-dot > out.dot 2>&1
#include <assert.h>

bool lex(const char *s) {
    const char *YYCURSOR = s;
    /*!re2c
        re2c:yyfill:enable = 0;
        re2c:define:YYCTYPE = char;

        number = (#aa[a])+[cdg]+;

        number { return true; }
    */
}

// number = [b]|(((#gg[a])|[?]){2})?;
// number = (([abe]*)[abcd]([cdh]*));
// number = (([ab\u0fa0-\u1388]*)[abcd]([cdh]*));
// number = ([^a-z]|"k");
// number = (("a"*)[ab]("b"*));
// number = (("a"|"b"|"e") | ("a"|"b"|"c"));
// number = ((#a("a"|"b"|"c")){3,});
// number = ((#a"a"){2});
// number = (#a"a")*#b("a"|"f")"f"*;
// number = (((#a"a"|"c"){2})|"d");
// number = (@a"a")*@b("a"|"f")"f"*;
// number = (@o"a"|"?"){2};
// number = "b"|(!("a")|"?"){2}?"f"*;

int main() {
    assert(lex("1234"));
    return 0;
}
