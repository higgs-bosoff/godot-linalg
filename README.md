# godot-linalg
Linear Algebra library in GDScript for Godot Engine

All methods are optimised for maximum speed. They take arrays and assume the 
right dimension for them. If the inputs aren't right they'll crash. Third input
is used for the answer, preallocated. There are no conditional branchings. 
Just use the method appropriate to the situation. The names are coded to reflect that. s = scalar, v = vector and m = matrix. So for example

    dot_vm(v, M)

is a dot product between vector and matrix (in that order). Wherever the `in_place` argument is provided, it is possible to perform the operation on the object itself instead of instantiating a new one (this too optimises performance). So for example
    
    transpose(M, true)

will turn M into its own transpose by reference, whereas

     MT = transpose(M)

will leave M unaltered.

Most method names are self-explanatory. The less intuitive are:

* `householder(v)`: computes the [Householder matrix](https://en.wikipedia.org/wiki/Householder_transformation) of a vector;
* `qr(M)`: computes the [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition) of a square matrix;
* `eigs_powerit`: computes the eigenvalues and eigenvectors of a square symmetric matrix using [the power iteration algorithm](https://en.wikipedia.org/wiki/Power_iteration) combined with a Hotelling deflation (check out [this book](https://www-users.cs.umn.edu/~saad/eig_book_2ndEd.pdf) for more details).

