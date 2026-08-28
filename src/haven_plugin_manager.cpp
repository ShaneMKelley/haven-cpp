#include "haven/haven_plugin_manager.h"
#include <iostream>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace haven {

PluginManager::PluginManager() = default;

PluginManager::~PluginManager() {
    unload_all();
}

size_t PluginManager::discover_plugins(const std::string& plugins_directory) {
    if (!std::filesystem::exists(plugins_directory)) {
        try {
            std::filesystem::create_directories(plugins_directory);
        } catch (...) {}
        return 0;
    }

    size_t count = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(plugins_directory)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
#ifdef _WIN32
            if (ext == ".dll") {
                if (load_plugin_file(entry.path().string())) {
                    count++;
                }
            }
#else
            if (ext == ".so") {
                if (load_plugin_file(entry.path().string())) {
                    count++;
                }
            }
#endif
        }
    }
    return count;
}

bool PluginManager::load_plugin_file(const std::string& filepath, PluginExecutionContext* ctx) {
#ifdef _WIN32
    HMODULE hModule = LoadLibraryA(filepath.c_str());
    if (!hModule) {
        std::cerr << "⚠️ Failed to load plugin library: " << filepath << " (Error " << GetLastError() << ")\n";
        return false;
    }

    auto create_func = reinterpret_cast<CreatePluginFunc>(GetProcAddress(hModule, "create_haven_plugin"));
    auto destroy_func = reinterpret_cast<DestroyPluginFunc>(GetProcAddress(hModule, "destroy_haven_plugin"));

    if (!create_func) {
        std::cerr << "⚠️ Plugin library missing 'create_haven_plugin' entrypoint: " << filepath << "\n";
        FreeLibrary(hModule);
        return false;
    }

    IHavenPlugin* raw_instance = create_func();
    if (!raw_instance) {
        std::cerr << "⚠️ Plugin factory returned null instance: " << filepath << "\n";
        FreeLibrary(hModule);
        return false;
    }

    PluginMetadata meta = raw_instance->get_metadata();
    if (!raw_instance->on_load(ctx)) {
        std::cerr << "⚠️ Plugin failed on_load initialization: " << meta.name << "\n";
        if (destroy_func) destroy_func(raw_instance);
        else delete raw_instance;
        FreeLibrary(hModule);
        return false;
    }

    LoadedPluginRecord record;
    record.filepath = filepath;
    record.metadata = meta;
    record.instance = std::shared_ptr<IHavenPlugin>(raw_instance, [destroy_func](IHavenPlugin* p) {
        if (p) {
            p->on_unload();
            if (destroy_func) destroy_func(p);
            else delete p;
        }
    });
    record.dll_handle = hModule;
    record.destroy_func = destroy_func;

    plugins_[meta.id] = record;
    std::cout << "✨ [Plugin Loaded] " << meta.name << " v" << meta.version << " by " << meta.author << " (" << meta.description << ")\n";
    return true;
#else
    void* handle = dlopen(filepath.c_str(), RTLD_NOW);
    if (!handle) {
        std::cerr << "⚠️ Failed to dlopen plugin library: " << filepath << " (" << dlerror() << ")\n";
        return false;
    }

    auto create_func = (CreatePluginFunc)dlsym(handle, "create_haven_plugin");
    auto destroy_func = (DestroyPluginFunc)dlsym(handle, "destroy_haven_plugin");

    if (!create_func) {
        std::cerr << "⚠️ Plugin missing 'create_haven_plugin': " << filepath << "\n";
        dlclose(handle);
        return false;
    }

    IHavenPlugin* raw_instance = create_func();
    if (!raw_instance) {
        dlclose(handle);
        return false;
    }

    PluginMetadata meta = raw_instance->get_metadata();
    if (!raw_instance->on_load(ctx)) {
        if (destroy_func) destroy_func(raw_instance);
        else delete raw_instance;
        dlclose(handle);
        return false;
    }

    LoadedPluginRecord record;
    record.filepath = filepath;
    record.metadata = meta;
    record.instance = std::shared_ptr<IHavenPlugin>(raw_instance, [destroy_func](IHavenPlugin* p) {
        if (p) {
            p->on_unload();
            if (destroy_func) destroy_func(p);
            else delete p;
        }
    });
    record.so_handle = handle;
    record.destroy_func = destroy_func;

    plugins_[meta.id] = record;
    std::cout << "✨ [Plugin Loaded] " << meta.name << " v" << meta.version << " (" << meta.description << ")\n";
    return true;
#endif
}

bool PluginManager::register_native_plugin(std::shared_ptr<IHavenPlugin> plugin, PluginExecutionContext* ctx) {
    if (!plugin) return false;
    PluginMetadata meta = plugin->get_metadata();
    if (!plugin->on_load(ctx)) return false;

    LoadedPluginRecord record;
    record.filepath = "[in-process]";
    record.metadata = meta;
    record.instance = plugin;
    plugins_[meta.id] = record;

    std::cout << "✨ [In-Process Plugin Registered] " << meta.name << " (" << meta.description << ")\n";
    return true;
}

bool PluginManager::unload_plugin(const std::string& plugin_id) {
    auto it = plugins_.find(plugin_id);
    if (it == plugins_.end()) return false;

    it->second.instance.reset(); // triggers custom deleter which calls on_unload()

#ifdef _WIN32
    if (it->second.dll_handle) {
        FreeLibrary(it->second.dll_handle);
    }
#else
    if (it->second.so_handle) {
        dlclose(it->second.so_handle);
    }
#endif

    plugins_.erase(it);
    return true;
}

void PluginManager::unload_all() {
    for (auto& [id, record] : plugins_) {
        record.instance.reset();
#ifdef _WIN32
        if (record.dll_handle) FreeLibrary(record.dll_handle);
#else
        if (record.so_handle) dlclose(record.so_handle);
#endif
    }
    plugins_.clear();
}

void PluginManager::dispatch_prompt_prefill(std::string& prompt, std::vector<uint32_t>& tokens) {
    for (auto& [id, record] : plugins_) {
        if (record.instance) {
            record.instance->on_prompt_prefill(prompt, tokens);
        }
    }
}

void PluginManager::dispatch_token_generated(uint32_t token_id, const std::string& piece) {
    for (auto& [id, record] : plugins_) {
        if (record.instance) {
            record.instance->on_token_generated(token_id, piece);
        }
    }
}

bool PluginManager::dispatch_tool_execution(const std::string& action, const std::string& payload, std::string& output) {
    for (auto& [id, record] : plugins_) {
        if (record.instance && record.instance->can_handle_tool(action)) {
            return record.instance->execute_tool(action, payload, output);
        }
    }
    return false;
}

void PluginManager::dispatch_sensory_tick(std::vector<float>& vision_emb, std::vector<float>& audio_emb) {
    for (auto& [id, record] : plugins_) {
        if (record.instance) {
            record.instance->on_sensory_tick(vision_emb, audio_emb);
        }
    }
}

size_t PluginManager::reload_all(const std::string& plugins_directory, PluginExecutionContext* ctx) {
    plugins_dir_ = plugins_directory;
    unload_all();
    return discover_plugins(plugins_dir_);
}

std::string PluginManager::get_tools_prompt_description() const {
    if (plugins_.empty()) return "";
    std::string s = "\n[Autonomous Sovereign Capabilities & Active Tools]:\n";
    for (const auto& [id, record] : plugins_) {
        s += " • " + record.metadata.name + ": " + record.metadata.description + "\n";
    }
    s += "Available tool actions: 'wiki_summary <topic>', 'get_weather <city>', 'yt_music_search <query>', 'yt_music_toggle', 'yt_music_next', 'get_active_window', 'screen_info', 'vault_search <query>', 'diary_write <entry>', 'system_telemetry'.\n";
    s += "To invoke a tool, write: <|tool_call|> tool_name payload <|tool_call|>\n";
    return s;
}

} // namespace haven
