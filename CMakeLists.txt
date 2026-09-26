cmake_minimum_required(VERSION 3.28.0...3.28.0 FATAL_ERROR)

project(OsteotomyCuts)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/OsteotomyCuts")
set(EXTENSION_CONTRIBUTORS "Manjula Herath (FaceLab.care)")
set(EXTENSION_DESCRIPTION "This is a extension to make OsteotomyCuts")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/OsteotomyCuts.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/OsteotomyCuts/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(OsteotomyCuts)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
