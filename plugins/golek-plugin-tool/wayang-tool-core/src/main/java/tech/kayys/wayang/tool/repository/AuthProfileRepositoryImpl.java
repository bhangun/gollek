package tech.kayys.wayang.tool.repository;

import io.quarkus.hibernate.reactive.panache.PanacheRepository;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.tool.utils.AuthProfile;

import java.util.List;

@ApplicationScoped
public class AuthProfileRepositoryImpl implements PanacheRepository<AuthProfile>, AuthProfileRepository {

    @Override
    public Uni<List<AuthProfile>> listAllProfiles() {
        return listAll();
    }

    @Override
    public Uni<List<AuthProfile>> findByTenantId(String tenantId) {
        return list("tenantId", tenantId);
    }

    @Override
    public Uni<List<AuthProfile>> findByTenantIdAndEnabled(String tenantId, boolean enabled) {
        return list("tenantId = ?1 AND enabled = ?2", tenantId, enabled);
    }

    @Override
    public Uni<AuthProfile> findById(String profileId) {
        return find("profileId", profileId).firstResult();
    }

    @Override
    public Uni<AuthProfile> findByTenantIdAndProfileId(String tenantId, String profileId) {
        return find("tenantId = ?1 AND profileId = ?2", tenantId, profileId).firstResult();
    }

    @Override
    public Uni<AuthProfile> save(AuthProfile profile) {
        return persist(profile);
    }

    @Override
    public Uni<AuthProfile> update(AuthProfile profile) {
        return persist(profile);
    }

    @Override
    public Uni<Boolean> deleteById(String profileId) {
        return delete("profileId", profileId).map(deleted -> deleted > 0);
    }

    @Override
    public Uni<List<AuthProfile>> searchProfiles(String query, Object... params) {
        return list(query, params);
    }

    @Override
    public Uni<Long> count() {
        return PanacheRepository.super.count();
    }

    @Override
    public Uni<Long> countByTenantId(String tenantId) {
        return count("tenantId", tenantId);
    }
}