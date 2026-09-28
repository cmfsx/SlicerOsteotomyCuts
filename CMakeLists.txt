# SPDX-FileCopyrightText: 2026 Manjula Herath
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of OsteotomyCuts, a 3D Slicer extension. See LICENSE and DISCLAIMER.md.

cmake_minimum_required(VERSION 3.28.0...3.28.0 FATAL_ERROR)

project(OsteotomyCuts)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/cmfsx/SlicerOsteotomyCuts")
set(EXTENSION_CATEGORY "Surgical Planning")
set(EXTENSION_CONTRIBUTORS "Manjula Herath (Ministry of Health, Sri Lanka; FaceLab, Colombo, Sri Lanka; Malmö University, Sweden)")
set(EXTENSION_DESCRIPTION "Virtual osteotomies for orthognathic and craniofacial surgical planning: draw osteotomy lines on a bone model, set the saw direction, blade thickness and cut depth, and divide the bone into closed bone segments, with checks of model quality and of the distance to the nerve canal and tooth roots. Research and planning software, not a medical device. Licensed under GPL-3.0-or-later.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/cmfsx/SlicerOsteotomyCuts/main/OsteotomyCuts.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/cmfsx/SlicerOsteotomyCuts/main/Screenshots/1.png https://raw.githubusercontent.com/cmfsx/SlicerOsteotomyCuts/main/Screenshots/2.png https://raw.githubusercontent.com/cmfsx/SlicerOsteotomyCuts/main/Screenshots/3.png")
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
