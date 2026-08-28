#pragma once

#include "haven_plugin.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <filesystem>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace haven {

struct LoadedPluginRecord {
    std::string filepath;
    PluginMetadata metadata;
    std::shared_ptr<IHavenPlugin> instance;
#ifdef _WIN32
    HMODULE dll_handle = nullptr;
#else
    void* so_handle = nullptr;
#endif
    DestroyPluginFunc destroy_func = nullptr;
};

class PluginManager {
public:
    PluginManager();
    ~PluginManager();

    // Auto-discovers and loads all plugins in the given directory
    size_t discover_plugins(const std::string& plugins_directory = "plugins");

    // Manually loads a single dynamic library plugin (.dll / .so)
    bool load_plugin_file(const std::string& filepath, PluginExecutionContext* ctx = nullptr);

    // Registers an in-process plugin directly
    bool register_native_plugin(std::shared_ptr<IHavenPlugin> plugin, PluginExecutionContext* ctx = nullptr);

    // Unloads a specific plugin by its ID
    bool unload_plugin(const std::string& plugin_id);

    // Unloads all active plugins safely
    void unload_all();

    // Plugin inspection
    const std::unordered_map<std::string, LoadedPluginRecord>& get_loaded_plugins() const { return plugins_; }
    size_t get_plugin_count() const { return plugins_.size(); }
    bool has_plugin(const std::string& plugin_id) const { return plugins_.find(plugin_id) != plugins_.end(); }

    // Hot-reloads all plugins from the directory without restarting engine
    size_t reload_all(const std::string& plugins_directory = "plugins", PluginExecutionContext* ctx = nullptr);

    // Returns formatted prompt describing all loaded tools and how Aura can call them
    std::string get_tools_prompt_description() const;

    // Dispatchers for engine lifecycle events
    void dispatch_prompt_prefill(std::string& prompt, std::vector<uint32_t>& tokens);
    void dispatch_token_generated(uint32_t token_id, const std::string& piece);
    bool dispatch_tool_execution(const std::string& action, const std::string& payload, std::string& output);
    void dispatch_sensory_tick(std::vector<float>& vision_emb, std::vector<float>& audio_emb);

private:
    std::unordered_map<std::string, LoadedPluginRecord> plugins_;
    std::string plugins_dir_ = "plugins";
};

} // namespace haven
