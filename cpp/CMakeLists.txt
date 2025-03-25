cmake_minimum_required(VERSION 3.22)
project(MagniForge VERSION 0.0.1)

# Set up a libs directory in your project (can put JUCE library and test libraries in here)
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs)

# Auto-download CPM if not already present
set(CPM_DOWNLOAD_VERSION 0.34.0)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if (NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif ()
include(${CPM_DOWNLOAD_LOCATION})

# Use CPM to fetch JUCE, specifying where to put it
CPMAddPackage(
    NAME JUCE
    GITHUB_REPOSITORY juce-framework/JUCE
    GIT_TAG 8.0.6  # Latest version as of March 2025
    SOURCE_DIR ${LIB_DIR}/juce
)

juce_add_plugin(MagniForge
    VERSION ${PROJECT_VERSION}
    COMPANY_NAME "Sembience"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
    PLUGIN_MANUFACTURER_CODE Semb
    PLUGIN_CODE MgFr
    FORMATS AU VST3 Standalone
    PRODUCT_NAME "Magni Forge")        # The name of the final executable, which can differ from the target name


target_sources(MagniForge
    PRIVATE
        Source/PluginEditor.cpp
        Source/PluginEditor.h
        Source/PluginProcessor.cpp
        Source/PluginProcessor.h
)

# Set source group for Xcode (for clean folder structure)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/Source PREFIX "Source" FILES
    Source/PluginEditor.cpp
    Source/PluginEditor.h
    Source/PluginProcessor.cpp
    Source/PluginProcessor.h
)

target_compile_definitions(MagniForge
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0)

# To add binary resources (images, fonts, etc.) to plugin:
# 1. Uncomment line below
# 2. Replace "..." with paths resource files (e.g., "Resources/logo.png" "Resources/knob.svg")
# 3. Add binary data target to target_link_libraries below (uncomment the MagniForgeData line)
# juce_add_binary_data(MagniForgeData SOURCES ...)

target_link_libraries(MagniForge
    PRIVATE
        # MagniForgeData           # Uncomment this when using binary resources
        juce::juce_audio_utils
        juce::juce_dsp
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)
