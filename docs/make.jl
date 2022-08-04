using Normalization
using Documenter

DocMeta.setdocmeta!(Normalization, :DocTestSetup, :(using Normalization); recursive=true)

makedocs(;
    modules=[Normalization],
    authors="brendanjohnharris <brendanjohnharris@gmail.com> and contributors",
    repo="https://github.com/brendanjohnharris/Normalization.jl/blob/{commit}{path}#{line}",
    sitename="Normalization.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
