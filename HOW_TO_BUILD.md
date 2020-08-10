# How to Build LinAlg GDNative plugin
Builds fine with mingw-gcc.

## Windows:
  - Get [Scoop](https://scoop.sh/).
  - `scoop install gcc`

## Any:
    Replace values in <> with the required values.
  - Compile `godot-cpp`: `cd godot-cpp; scons platform=<PLATFORM> generate_bindings=yes -j<NUMBER_OF_THREADS_TO_USE> bits=64 use_mingw=yes; cd ..`
  - Compile `liblinalg`: `scons platform=<PLATFORM> -j<NUMBER_OF_THREADS_TO_USE> bits=64 use_mingw=yes`

This build uses a modified `godot-cpp` submodule. If you clone it separately, make sure to change [this line in Defs.hpp](https://github.com/godotengine/godot-cpp/blob/cd69b58bb6a9c804e27054b6e29a136453a00f04/include/core/Defs.hpp#L65) (based on [this PR](https://github.com/godotengine/godot-cpp/pull/415)) to this:
```cpp
// alloca() is non-standard. When using MSVC, it's in malloc.h.
#if defined(__linux__) || defined(__APPLE__) //|| defined(__MINGW32__)
#include <alloca.h>
#endif
```
