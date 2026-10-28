#!/usr/bin/env bash
#
# Build, synchronize, execute, serve, and clean the Qamomile documentation.
#
# Usage:
#   ./docs/build.sh [COMMAND] [PAGE_SOURCE ...]
#   ./build.sh [COMMAND] [PAGE_SOURCE ...]  # when run from docs/
#
# Arguments:
#   COMMAND:
#     Documentation operation to perform. Defaults to "help". Run with
#     "help" to list every supported command.
#   PAGE_SOURCE:
#     One or more jupytext .py files accepted only by "page-build". Paths
#     must be under docs/en/ or docs/ja/ and may be relative to either the
#     repository root or docs/.
#
# Environment:
#   READTHEDOCS:
#     A value of "True" enables Read the Docs build behavior.
#   READTHEDOCS_CANONICAL_URL:
#     Any nonempty value also enables Read the Docs build behavior.
#   READTHEDOCS_VERSION:
#     When Read the Docs is detected, a nonempty value is used to construct
#     the language-specific BASE_URL passed to jupyter-book.
#   QAMOMILE_DOCS_TEST:
#     Inherited by executed notebooks. Leave unset or different from "1" when
#     producing reader-facing notebook outputs.
#
# Outputs:
#   Depending on COMMAND, generates API reference files, synchronizes or
#   executes notebooks, creates HTML builds, serves a local HTTP server, or
#   removes generated documentation artifacts. Progress and diagnostics are
#   written to the standard streams.
#
# Exit status:
#   0 when the requested command completes successfully.
#   Nonzero when the command is invalid, an input page is invalid, or an
#   invoked documentation tool fails.

set -e  # Exit on error

# Move to the script's directory to ensure relative paths work
cd "$(dirname "$0")"

# Languages and target directories
LANGS=(en ja)
# Sections whose .py sources are synced to .ipynb. This includes integration
# because jupytext sync only rewrites cell sources and does not execute code.
SYNC_DIRS=(tutorial algorithm usage integration)
# Sections whose notebooks are executed by bulk execute/sync-build targets.
# integration is excluded because some pages currently require credentials;
# docs tests cover integration pages with file-level skips for credentialed
# notebooks.
# release_notes is excluded because it is markdown-only; nothing to sync or execute.
TARGET_DIRS=(tutorial algorithm usage)

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print a green success message.
#
# Arguments:
#   $1: Message to print.
#
# Globals:
#   GREEN, NC: ANSI color sequences used to format the message.
#
# Outputs:
#   Writes the formatted message to stdout.
#
# Returns:
#   The exit status returned by echo.
info()  { echo -e "${GREEN}✓${NC} $1"; }

# Print a red error message and terminate the script.
#
# Arguments:
#   $1: Message to print.
#
# Globals:
#   RED, NC: ANSI color sequences used to format the message.
#
# Outputs:
#   Writes the formatted message to stdout.
#
# Exit status:
#   Always exits the script with status 1.
error() { echo -e "${RED}✗${NC} $1"; exit 1; }

# Print a yellow warning message.
#
# Arguments:
#   $1: Message to print.
#
# Globals:
#   YELLOW, NC: ANSI color sequences used to format the message.
#
# Outputs:
#   Writes the formatted message to stdout.
#
# Returns:
#   The exit status returned by echo.
warn()  { echo -e "${YELLOW}!${NC} $1"; }

# Display the command-line help text.
#
# Arguments:
#   None.
#
# Outputs:
#   Writes usage information and the supported commands to stdout.
#
# Returns:
#   The exit status returned by the final echo command.
show_help() {
    echo "Qamomile Documentation Build System"
    echo "===================================="
    echo ""
    echo "Usage: ./build.sh [command]"
    echo ""
    echo "Available commands:"
    echo "  build          - Build both languages (no source-tree sync or notebook execution)"
    echo "  build-en       - Build English only (no source-tree sync or notebook execution)"
    echo "  build-ja       - Build Japanese only (no source-tree sync or notebook execution)"
    echo "  sync           - Convert .py files under SYNC_DIRS to .ipynb (both languages)"
    echo "  sync-en        - Convert English .py files under SYNC_DIRS to .ipynb"
    echo "  sync-ja        - Convert Japanese .py files under SYNC_DIRS to .ipynb"
    echo "  execute        - Execute bulk-runnable .ipynb notebooks (both languages)"
    echo "  execute-en     - Execute English bulk-runnable .ipynb notebooks"
    echo "  execute-ja     - Execute Japanese bulk-runnable .ipynb notebooks"
    echo "  sync-build     - Sync, execute bulk-runnable notebooks, and build both languages"
    echo "  sync-build-en  - Sync, execute bulk-runnable notebooks, and build English documentation"
    echo "  sync-build-ja  - Sync, execute bulk-runnable notebooks, and build Japanese documentation"
    echo "  page-build     - Sync, execute, and build one or more page sources"
    echo "  clean          - Remove generated target notebooks, copied API docs, and build outputs"
    echo "  clean-all      - Run clean and remove execution cache"
    echo "  serve-en       - Serve English docs; sync-build first if output is missing (port 8000)"
    echo "  serve-ja       - Serve Japanese docs; sync-build first if output is missing (port 8000)"
    echo "  fresh-en       - Clean, sync-build, and serve English docs"
    echo "  fresh-ja       - Clean, sync-build, and serve Japanese docs"
    echo "  help           - Show this help message"
    echo ""
}

# Return success when the environment indicates a Read the Docs build.
#
# Arguments:
#   None.
#
# Environment:
#   READTHEDOCS: Recognized when equal to "True".
#   READTHEDOCS_CANONICAL_URL: Recognized when nonempty.
#
# Returns:
#   0 when either Read the Docs indicator is present.
#   1 otherwise.
is_rtd() {
    [ "${READTHEDOCS:-}" = "True" ] || [ -n "${READTHEDOCS_CANONICAL_URL:-}" ]
}

# Generate the API reference from Qamomile docstrings.
#
# Arguments:
#   None.
#
# Outputs:
#   Recreates the generated API reference under docs/api/, refreshes the
#   managed API TOC region in docs/en/myst.yml and docs/ja/myst.yml, and writes
#   progress messages to stdout.
#
# Returns:
#   0 on successful completion.
generate_api() {
    echo "Generating API reference..."
    uv run python generate_api.py
    info "API reference generated"
}

# Generate tag indexes and inject auto-managed tag content into a docs tree.
#
# Arguments:
#   $1: Optional docs root to process. Defaults to the current directory.
#
# Outputs:
#   Updates tag-related content below the selected docs root and writes
#   progress messages to stdout.
#
# Returns:
#   0 on successful completion.
generate_doc_tags() {
    # Inject auto-managed regions (chip blocks, browse-by-tag clouds,
    # per-tag pages) into the build-dir copy of the docs tree pointed
    # at by $1. Defaults to the current dir when invoked without an
    # argument (back-compat for ad-hoc runs).
    local docs_root="${1:-$(pwd)}"
    echo "Generating doc tag indexes (docs root: ${docs_root})..."
    DOCS_ROOT_OVERRIDE="$docs_root" uv run python scripts/build_doc_tags.py
    info "Doc tag pages generated"
}

# Prepare an isolated documentation source tree for one or more languages.
#
# Arguments:
#   $@: Optional locale names to copy. When omitted, every locale in LANGS is
#       prepared.
#
# Globals:
#   LANGS: Default locales prepared when no arguments are supplied.
#   SYNC_DIRS: Sections whose jupytext sources are updated in the scratch tree.
#
# Outputs:
#   Recreates docs/_build_src/, copies the selected language trees and shared
#   assets, injects tag content, and updates paired notebooks in the scratch
#   tree. Writes progress messages to stdout.
#
# Returns:
#   0 on successful completion.
setup_build_src() {
    # Build everything inside docs/_build_src/ so the committed source
    # tree never receives auto-managed injections. Sequence:
    #   1. rm + recreate docs/_build_src/<lang>/ as a copy of docs/<lang>/
    #   2. run build_doc_tags.py against _build_src/ (injects chip
    #      blocks and browse-by-tag clouds; generates per-tag pages)
    #   3. jupytext --update so chip-injected .py propagates into .ipynb
    #      cells (preserves committed outputs). Includes integration/
    #      because jupytext only syncs cell sources — no execution, no
    #      API-key dependency.
    #
    # Pass one or more locale names as arguments to scope the work to
    # those locales (used by build_lang); pass no arguments to set up
    # every locale in $LANGS (used by build_all).
    local langs=("$@")
    if [ ${#langs[@]} -eq 0 ]; then
        langs=("${LANGS[@]}")
    fi
    local build_src_root="$(pwd)/_build_src"
    local _lang _dir _py_files
    echo "Setting up _build_src/ for: ${langs[*]} ..."
    rm -rf "$build_src_root"
    mkdir -p "$build_src_root"
    for _lang in "${langs[@]}"; do
        # cp -R is portable across the Linux RTD image and local macOS
        # (rsync isn't installed in the RTD build env). Copy the whole
        # lang dir then prune the build artifacts that we never want
        # in the scratch tree.
        cp -R "${_lang}" "${build_src_root}/"
        rm -rf "${build_src_root}/${_lang}/_build"
    done
    # myst.yml inside each <lang>/ references ../assets/custom-theme.css
    # (the chip-pill CSS, the logo, the colab-launch JS). When mystmd
    # builds from _build_src/<lang>/, that ../assets/ resolves to
    # _build_src/assets/, so the assets dir has to exist there too.
    cp -R assets "${build_src_root}/"

    # build_doc_tags.py auto-skips locales whose root dir is absent
    # from the build-dir, so a single-locale invocation only does work
    # for the locales we actually copied above.
    generate_doc_tags "$build_src_root"

    for _lang in "${langs[@]}"; do
        for _dir in "${SYNC_DIRS[@]}"; do
            shopt -s nullglob
            _py_files=("${build_src_root}/${_lang}/${_dir}"/*.py)
            shopt -u nullglob
            [ ${#_py_files[@]} -eq 0 ] && continue
            uv run jupytext --to ipynb --update "${_py_files[@]}"
        done
    done

    info "_build_src/ ready (${langs[*]})"
}

# Copy the generated API reference into every language directory.
#
# Arguments:
#   None.
#
# Globals:
#   LANGS: Locales that receive a copied API reference.
#
# Outputs:
#   Creates or updates docs/<lang>/api/ for every locale in LANGS and writes
#   progress messages to stdout.
#
# Returns:
#   0 on successful completion.
copy_api() {
    # NOTE: ``local _lang`` is mandatory. Without it the loop variable
    # would leak into the caller's scope and clobber the caller's own
    # ``local lang="$1"`` (build_lang sets that, then calls copy_api,
    # then calls setup_build_src "$lang" — which would receive the
    # wrong locale once copy_api's loop finishes on the last LANGS
    # element). Same trap exists in clean / clean_all / execute_lang.
    local _lang
    echo "Copying API reference to language directories..."
    for _lang in "${LANGS[@]}"; do
        mkdir -p "${_lang}/api"
        cp -r api/. "${_lang}/api/"
    done
    info "API reference copied"
}

# Synchronize jupytext .py sources to .ipynb files for one language.
#
# Arguments:
#   $1: Locale directory name, such as "en" or "ja".
#
# Globals:
#   SYNC_DIRS: Sections whose .py sources are converted.
#
# Outputs:
#   Creates or updates paired notebooks in every SYNC_DIRS section for the
#   selected locale and writes progress or warning messages to stdout.
#
# Returns:
#   0 on successful completion.
sync_lang() {
    local lang="$1"
    echo "Converting ${lang} .py files to .ipynb..."
    for dir in "${SYNC_DIRS[@]}"; do
        local py_files=()
        shopt -s nullglob
        py_files=("${lang}/${dir}"/*.py)
        shopt -u nullglob
        if [ ${#py_files[@]} -eq 0 ]; then
            warn "No .py files in ${lang}/${dir}, skipping"
            continue
        fi
        uv run jupytext --to ipynb "${py_files[@]}"
    done
    # execute_lang still excludes integration/ because this shell command
    # has no file-level skip list for credentialed notebooks.
    info "${lang} notebooks synced"
}

# Execute bulk-runnable .ipynb notebooks for one language.
#
# Arguments:
#   $1: Locale directory name, such as "en" or "ja".
#
# Environment:
#   QAMOMILE_DOCS_TEST: Inherited by notebook execution; leave unset or
#   different from "1" for reader-facing outputs.
#
# Globals:
#   TARGET_DIRS: Sections whose notebooks are executed.
#
# Outputs:
#   Rewrites each existing notebook directly under the selected locale's
#   TARGET_DIRS sections with executed outputs and writes progress messages to
#   stdout.
#
# Returns:
#   0 on successful completion.
execute_lang() {
    local lang="$1"
    local dir nb
    echo "Executing ${lang} notebooks..."
    for dir in "${TARGET_DIRS[@]}"; do
        for nb in "${lang}/${dir}"/*.ipynb; do
            [ -f "$nb" ] || continue
            info "Executing ${nb}..."
            uv run jupyter nbconvert --to notebook --execute --inplace "$nb"
        done
    done
    info "${lang} notebooks executed"
}

# Build one language from docs/_build_src/<lang>/.
#
# Arguments:
#   $1: Locale directory name, such as "en" or "ja".
#
# Environment:
#   READTHEDOCS, READTHEDOCS_CANONICAL_URL:
#     Select Read the Docs build behavior through is_rtd.
#   READTHEDOCS_VERSION:
#     When nonempty on Read the Docs, forms the BASE_URL for this locale.
#
# Outputs:
#   Builds HTML in the scratch tree, replaces docs/<lang>/_build/ with that
#   build, injects Colab launch controls, and writes progress messages to
#   stdout.
#
# Preconditions:
#   setup_build_src must already have prepared the selected locale.
#
# Returns:
#   0 on successful completion.
_build_lang_from_build_src() {
    local lang="$1"
    echo "Building ${lang} documentation..."
    cd "_build_src/${lang}"
    if is_rtd && [[ -n "${READTHEDOCS_VERSION:-}" ]]; then
        local base_url="/${READTHEDOCS_VERSION}/${lang}"
        info "Read the Docs detected. Using BASE_URL=${base_url}"
        BASE_URL="$base_url" uv run jupyter-book build --html
    else
        if is_rtd; then
            warn "Read the Docs detected but READTHEDOCS_VERSION is unset or empty; building without BASE_URL"
        fi
        uv run jupyter-book build --html
    fi
    cd ../..
    # Move the html output back to docs/<lang>/_build/ so the existing
    # post-build step (colab-launch injection) and the .readthedocs.yaml
    # copy step both find it where they used to.
    rm -rf "${lang}/_build"
    cp -r "_build_src/${lang}/_build" "${lang}/_build"
    uv run python scripts/inject_colab_launch.py "$lang"
    info "${lang} documentation built: ${lang}/_build/html/index.html"
}

# Build documentation for a single language with no source-tree sync or
# notebook execution. Public entry point — runs generate_api + copy_api +
# setup_build_src then builds just this lang. We always run the API
# generation pair so the build is self-contained: a contributor running
# ``./build.sh build-en`` on a fresh clone (where ``docs/api/`` is
# gitignored and absent) does not get a missing-toc-entry error from
# mystmd. The pair is fast and idempotent, so re-running them on every
# single-locale build is an acceptable cost; ``build_all`` calls them
# once up front and then delegates to ``_build_lang_from_build_src``
# directly so we don't double-run.
#
# Arguments:
#   $1: Locale directory name, such as "en" or "ja".
#
# Outputs:
#   Regenerates docs/api/ and both managed API TOC regions, refreshes every
#   language's copied API files, prepares the selected locale in
#   docs/_build_src/, and replaces that locale's HTML build. Does not execute
#   committed notebooks.
#
# Returns:
#   0 on successful completion.
build_lang() {
    local lang="$1"
    generate_api
    copy_api
    setup_build_src "$lang"
    _build_lang_from_build_src "$lang"
}

# Synchronize, execute, and build documentation for one language.
#
# Arguments:
#   $1: Locale directory name, such as "en" or "ja".
#
# Outputs:
#   Updates the selected locale's paired notebooks and executed outputs,
#   regenerates the API source and TOCs, refreshes copied API files for every
#   locale in LANGS, and replaces the selected locale's HTML build.
#
# Returns:
#   0 on successful completion.
sync_build_lang() {
    local lang="$1"
    sync_lang "$lang"
    execute_lang "$lang"
    build_lang "$lang"
}

# Synchronize, execute, and build both English and Japanese documentation.
#
# Arguments:
#   None.
#
# Globals:
#   LANGS: Locales whose API copies and scratch sources are refreshed.
#
# Outputs:
#   Updates paired notebooks and executed outputs for both locales, regenerates
#   the API source, TOCs, copied API files, and scratch sources once, and
#   replaces both HTML builds.
#
# Returns:
#   0 on successful completion.
sync_build_all() {
    # Mirror build_all's pattern: run generate_api + copy_api ONCE up
    # front, then drive the per-locale work via the lower-level
    # primitives (sync_lang / execute_lang / setup_build_src /
    # _build_lang_from_build_src) instead of going through the
    # self-contained build_lang. Going through build_lang in a loop
    # would re-run generate_api + copy_api for every locale, which
    # noticeably slows ./build.sh sync-build for no benefit.
    generate_api
    copy_api
    sync_lang en
    sync_lang ja
    execute_lang en
    execute_lang ja
    # All committed source is now in sync; copy both locales into the
    # build-dir in a single setup_build_src call so the chip injection
    # and tag-page generation happen once across both locales.
    setup_build_src
    _build_lang_from_build_src en
    _build_lang_from_build_src ja
    info "Both English and Japanese documentation synced and built successfully"
}

# Validate and normalize one page-build source path.
#
# Arguments:
#   $1: Candidate path to a jupytext .py source below docs/en/ or docs/ja/.
#
# Outputs:
#   Writes the docs-relative path to stdout on success or a validation message
#   to stderr on failure.
#
# Returns:
#   0 when the path names an existing supported page source.
#   1 when the path is outside a supported locale, is not a .py file, or does
#   not exist.
normalize_page_source() {
    local raw="$1"
    local path="${raw#./}"
    path="${path#docs/}"
    if [[ "$path" != en/* && "$path" != ja/* ]]; then
        echo "Page source must be under docs/en/ or docs/ja/: ${raw}" >&2
        return 1
    fi
    if [[ "$path" != *.py ]]; then
        echo "Page source must be a jupytext .py file: ${raw}" >&2
        return 1
    fi
    if [ ! -f "$path" ]; then
        echo "Page source not found: ${raw}" >&2
        return 1
    fi
    echo "$path"
}

# Synchronize and execute selected pages, then build their languages.
#
# Arguments:
#   $@: One or more jupytext .py paths accepted by normalize_page_source.
#
# Environment:
#   QAMOMILE_DOCS_TEST: Inherited by notebook execution; leave unset or
#   different from "1" for reader-facing outputs.
#
# Outputs:
#   Updates and executes the paired notebook for every requested page,
#   regenerates the API source and TOCs, refreshes copied API files for every
#   locale in LANGS, prepares scratch sources for the affected locales, and
#   replaces their HTML builds.
#
# Exit status:
#   0 after every requested page and affected locale is built successfully.
#   1 when no page is supplied or a page fails validation.
#   Nonzero when an invoked documentation tool fails.
page_build() {
    if [ "$#" -eq 0 ]; then
        error "Usage: ./build.sh page-build docs/en/<section>/foo.py [docs/ja/<section>/foo.py ...]"
    fi

    local pages=()
    local raw page ipynb lang
    local build_en=0
    local build_ja=0
    for raw in "$@"; do
        page="$(normalize_page_source "$raw")" || exit 1
        pages+=("$page")
        lang="${page%%/*}"
        if [ "$lang" = "en" ]; then
            build_en=1
        else
            build_ja=1
        fi
    done

    for page in "${pages[@]}"; do
        ipynb="${page%.py}.ipynb"
        info "Syncing ${page} -> ${ipynb}..."
        uv run jupytext --to ipynb --update "$page"
        info "Executing ${ipynb}..."
        uv run jupyter nbconvert --to notebook --execute --inplace "$ipynb"
    done

    local langs=()
    [ "$build_en" -eq 1 ] && langs+=(en)
    [ "$build_ja" -eq 1 ] && langs+=(ja)
    generate_api
    copy_api
    setup_build_src "${langs[@]}"
    for lang in "${langs[@]}"; do
        _build_lang_from_build_src "$lang"
    done
    info "Requested page source(s) synced, executed, and built"
}

# Serve one language's HTML documentation on localhost port 8000.
#
# Arguments:
#   $1: Locale directory name, such as "en" or "ja".
#
# Outputs:
#   If the HTML build is missing, synchronizes, executes, and builds the
#   selected locale first, including refreshing copied API files for every
#   locale in LANGS. Changes the working directory to the selected HTML build,
#   writes the server URL to stdout, and blocks while serving on port 8000.
#
# Returns:
#   The exit status returned by the HTTP server when it stops.
serve_lang() {
    local lang="$1"
    if [ ! -d "${lang}/_build/html" ]; then
        warn "${lang} documentation not built. Syncing and building now..."
        generate_api
        copy_api
        sync_build_lang "$lang"
    fi
    echo "Serving ${lang} documentation at http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
    cd "${lang}/_build/html"
    uv run python -m http.server 8000
}

# Clean generated documentation, rebuild one language, and serve it.
#
# Arguments:
#   $1: Locale directory name, such as "en" or "ja".
#
# Outputs:
#   Removes generated artifacts for every locale, synchronizes, executes, and
#   rebuilds the selected locale, and refreshes copied API files for every
#   locale in LANGS. Changes the working directory to the selected HTML build
#   and blocks while serving on port 8000.
#
# Returns:
#   The exit status returned by the HTTP server when it stops.
fresh_lang() {
    local lang="$1"
    clean
    generate_api
    copy_api
    sync_build_lang "$lang"
    echo "Serving ${lang} documentation at http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
    cd "${lang}/_build/html"
    uv run python -m http.server 8000
}

# Build both languages without executing committed notebooks.
#
# Arguments:
#   None.
#
# Globals:
#   LANGS: Locales whose copied API files and scratch sources are refreshed.
#
# Outputs:
#   Regenerates the API source and TOCs, refreshes copied API files, prepares
#   both locales in the scratch source tree, and replaces both HTML builds.
#
# Returns:
#   0 on successful completion.
build_all() {
    generate_api
    copy_api
    setup_build_src
    _build_lang_from_build_src en
    _build_lang_from_build_src ja
    info "Both English and Japanese documentation built successfully"
}

# Remove generated documentation artifacts.
#
# Arguments:
#   None.
#
# Globals:
#   LANGS: Locales whose generated artifacts are removed.
#   TARGET_DIRS: Sections whose directly contained notebooks are removed.
#
# Outputs:
#   Removes notebooks directly under TARGET_DIRS, copied API references, HTML
#   builds, and docs/_build_src/ for every locale. Writes progress messages to
#   stdout.
#
# Returns:
#   0 on successful completion.
clean() {
    local _lang _dir
    echo "Cleaning generated files..."
    for _lang in "${LANGS[@]}"; do
        for _dir in "${TARGET_DIRS[@]}"; do
            rm -f "${_lang}/${_dir}"/*.ipynb
        done
        rm -rf "${_lang}/_build"
        rm -rf "${_lang}/api"
    done
    rm -rf "_build_src"
    info "Cleaned generated target notebooks, copied API docs, and build outputs"
}

# Remove generated documentation artifacts and defensively clear cache paths.
#
# Arguments:
#   None.
#
# Globals:
#   LANGS: Locales whose cache paths are checked after cleanup.
#
# Outputs:
#   Performs clean, which removes each locale's entire build directory, then
#   idempotently removes the now-normally-absent Jupyter cache paths. Writes
#   progress messages to stdout.
#
# Returns:
#   0 on successful completion.
clean_all() {
    local _lang
    clean
    echo "Cleaning execution cache..."
    for _lang in "${LANGS[@]}"; do
        rm -rf "${_lang}/_build/.jupyter_cache"
    done
    info "Generated target notebooks, copied API docs, build outputs, and execution cache removed"
}

# Main command dispatcher
case "${1:-help}" in
    build)      build_all ;;
    build-en)   build_lang en ;;
    build-ja)   build_lang ja ;;
    sync)       sync_lang en; sync_lang ja
                info "All SYNC_DIRS Python scripts converted to notebooks" ;;
    sync-en)    sync_lang en ;;
    sync-ja)    sync_lang ja ;;
    execute)    execute_lang en; execute_lang ja
                info "All bulk-runnable notebooks executed" ;;
    execute-en) execute_lang en ;;
    execute-ja) execute_lang ja ;;
    sync-build)     sync_build_all ;;
    sync-build-en)  sync_build_lang en ;;
    sync-build-ja)  sync_build_lang ja ;;
    page-build) shift; page_build "$@" ;;
    clean)      clean ;;
    clean-all)  clean_all ;;
    serve-en)   serve_lang en ;;
    serve-ja)   serve_lang ja ;;
    fresh-en)   fresh_lang en ;;
    fresh-ja)   fresh_lang ja ;;
    help|--help|-h) show_help ;;
    *)          error "Unknown command: $1. Run './build.sh help' for available commands." ;;
esac
